# CarGame DQN Training with Lane Following
# ========================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from collections import namedtuple, deque
import os
import pickle
import copy
from IPython.display import display, clear_output
import math

# Ensure model directory exists
os.makedirs("./car_dqn_models", exist_ok=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Car Physics Parameters (matching your C++ implementation)
CAR_PARAMS = {
    'width': 1.8,
    'length': 4.5,
    'height': 1.5,
    'wheel_base': 2.8,
    'front_axle_distance': 1.6,
    'rear_axle_distance': 1.2,
    'min_speed': -1.0,
    'max_steering_angle': 0.6,
    'steering_speed': 2.0,
    'min_movement_speed': 0.5,
    'gear_ratio': 7,
    'tire_radius': 0.33,
    'inertia_at_engine': 0.45,
    'mass': 1600,
    'cornering_stiffness_front': 50000.0,
    'cornering_stiffness_rear': 40000.0,
    'max_engine_speed': 8000.0,
    'idle_rpm': 1000.0,
}

# Lane parameters
LANE_PARAMS = {
    'lane_width': 4.0,  # meters
    'max_lateral_deviation': 2.5,  # meters (when to end episode)
    'lateral_deviation_penalty': 1.0  # reward penalty factor for lane deviation
}

# Car Simulation Class
class CarSimulation:
    def __init__(self, time_step=1.0/60.0):
        self.position = np.array([0.0, 0.5, 0.0])  # x, y, z
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.speed = 0.0
        self.rotation = 0.0
        self.steering_angle = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.engine_speed = CAR_PARAMS['idle_rpm']
        self.engine_speed_dot = 0.0
        self.wheel_rotation_speed = 0.0
        self.clutch = True
        self.time_step = time_step
        self.params = CAR_PARAMS
        
        # Lane following properties
        self.lateral_deviation = 0.0
        self.lane_heading_error = 0.0

    def reset(self, random_init=True):
        self.position = np.array([0.0, 0.5, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.speed = 0.0
        self.rotation = 0.0
        self.steering_angle = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.engine_speed = CAR_PARAMS['idle_rpm']
        self.engine_speed_dot = 0.0
        self.wheel_rotation_speed = 0.0
        self.clutch = True
        
        # Randomize initial conditions for better training
        if random_init:
            self.speed = np.random.uniform(0, 5.0)
            self.velocity[0] = 0  # No lateral velocity initially
            self.velocity[2] = self.speed  # Forward velocity
            
            # Random initial lateral position (lane deviation)
            self.position[0] = np.random.uniform(-1.0, 1.0)
            self.lateral_deviation = self.position[0]
            
            # Small random initial rotation
            self.rotation = np.random.uniform(-0.1, 0.1)
            self.lane_heading_error = self.rotation

    def get_engine_torque(self, throttle, rpm):
        # Model engine torque curve similar to your C++ implementation
        return throttle * (400.0 + 250.0 * (rpm / 4000.0) * (1.0 - rpm / 8000.0))

    def calculate_slip_ratio(self, wheel_linear_speed, vehicle_speed):
        # Calculate tire slip ratio for traction
        speed_abs = abs(vehicle_speed)
        min_speed = 0.5

        if speed_abs < min_speed:
            result = (wheel_linear_speed - vehicle_speed) / min_speed
        else:
            result = (wheel_linear_speed - vehicle_speed) / speed_abs

        return np.clip(result, -1.0, 1.0)

    def calculate_tire_force(self, slip_ratio):
        # Pacejka tire model for longitudinal forces
        D = 1.0
        C = 1.5
        B = 10.0
        E = 0.1
        Fz = 4000.0

        coefficient = D * np.sin(C * np.arctan(B * slip_ratio - E * (B * slip_ratio - np.arctan(B * slip_ratio))))
        return coefficient * Fz

    def calculate_lateral_tire_force(self, slip_angle, is_front_tire=True):
        # Pacejka tire model for lateral forces
        B = 10.0
        C = 1.3
        D = 1.0
        E = 1.0
        Fz = 4000.0 if is_front_tire else 3800.0
        
        argument = B * slip_angle - E * (B * slip_angle - np.arctan(B * slip_angle))
        peak = D * Fz
        
        lateral_force = peak * np.sin(C * np.arctan(argument))
        return -lateral_force

    def get_resistance_forces(self):
        # Calculate drag and rolling resistance
        air_density = 1.225
        drag_coefficient = 0.3
        frontal_area = self.params['width'] * self.params['height'] * 0.8

        drag_force = 0.5 * air_density * drag_coefficient * frontal_area * self.speed * abs(self.speed)

        rolling_coefficient = 0.015 * (1.0 + abs(self.speed) * 0.01)
        rolling_resistance = rolling_coefficient * self.params['mass'] * 9.81

        if self.speed != 0.0:
            rolling_resistance *= 1.0 if self.speed > 0.0 else -1.0

        return drag_force + rolling_resistance
    
    def update_longitudinal_physics(self):
        # Longitudinal vehicle dynamics
        idle_rpm = self.params['idle_rpm']

        if self.throttle > 0.1:
            self.clutch = False
            self.engine_speed_dot = self.get_engine_torque(self.throttle, self.engine_speed) / self.params['inertia_at_engine']
            self.engine_speed += self.engine_speed_dot * self.time_step
            self.engine_speed = min(self.engine_speed, self.params['max_engine_speed'])
        elif abs(self.speed) < 0.5:
            self.engine_speed = idle_rpm
            self.clutch = True
        else:
            wheel_rpm = abs(self.speed) / self.params['tire_radius'] * self.params['gear_ratio'] * (60.0 / (2.0 * math.pi))
            self.engine_speed = max(idle_rpm, wheel_rpm)

        if not self.clutch:
            if self.engine_speed > self.params['max_engine_speed']:
                self.engine_speed = self.params['max_engine_speed']

            engine_torque = self.get_engine_torque(self.throttle, self.engine_speed)

            if self.throttle < 0.1 and self.engine_speed > 1000.0:
                engine_torque -= (self.engine_speed / 8000.0) * 75.0

            self.wheel_rotation_speed = self.engine_speed / self.params['gear_ratio'] * (2.0 * math.pi / 60.0) if not self.clutch else self.wheel_rotation_speed

            brake_torque = 0.0
            if self.brake > 0.0:
                brake_torque = self.brake * 15000.0

                if self.wheel_rotation_speed > 0.0:
                    brake_torque = -brake_torque

            wheel_inertia = 5.0
            wheel_angular_accel = (engine_torque + brake_torque) / wheel_inertia

            self.wheel_rotation_speed += wheel_angular_accel * self.time_step
            wheel_linear_speed = self.wheel_rotation_speed * self.params['tire_radius']

            slip_ratio = self.calculate_slip_ratio(wheel_linear_speed, self.speed)
            longitudinal_force = 2 * self.calculate_tire_force(slip_ratio)

            total_resistance = self.get_resistance_forces()
            net_force = longitudinal_force - total_resistance

            self.acceleration[0] = 0  # Reset lateral acceleration
            self.acceleration[2] = net_force / self.params['mass']  # Longitudinal acceleration
            speed_change = self.acceleration[2] * self.time_step
            
            # Update speed (magnitude of velocity)
            self.speed += speed_change
    
    def update_lateral_physics(self):
        # Simple bicycle model for lateral dynamics
        MIN_SPEED = 2.0  # Minimum speed for meaningful lateral dynamics
        
        if abs(self.speed) < MIN_SPEED:
            # Simple steering at low speeds
            turn_rate = self.steering_angle * abs(self.speed) / self.params['wheel_base']
            self.rotation += turn_rate * self.time_step
            return
        
        # Calculate simplified slip angles
        lateral_velocity = self.velocity[0]
        yaw_rate = self.steering_angle * self.speed / self.params['wheel_base']
        
        # Update heading
        self.rotation += yaw_rate * self.time_step
        
        # Normalize rotation to -PI to PI
        while self.rotation > math.pi:
            self.rotation -= 2 * math.pi
        while self.rotation < -math.pi:
            self.rotation += 2 * math.pi
        
        # Calculate lane heading error (assuming lane is along z-axis)
        lane_direction = 0.0  # Lane direction is along z-axis
        self.lane_heading_error = self.rotation - lane_direction
        
        # Calculate lateral acceleration (simplified)
        lateral_accel = self.steering_angle * self.speed * self.speed / self.params['wheel_base']
        self.acceleration[0] = lateral_accel
        
        # Update lateral velocity
        lateral_velocity_change = lateral_accel * self.time_step
        self.velocity[0] += lateral_velocity_change
        
        # Apply damping to lateral velocity
        self.velocity[0] *= 0.95  # Simple damping factor
    
    def update_position(self):
        # Update velocity components based on heading
        self.velocity[0] = self.velocity[0]  # Lateral velocity already updated
        self.velocity[2] = self.speed * math.cos(self.rotation)  # Forward velocity
        
        # Update position
        self.position[0] += self.velocity[0] * self.time_step
        self.position[2] += self.velocity[2] * self.time_step
        
        # Update lateral deviation (position[0] is lateral position)
        self.lateral_deviation = self.position[0]

    def update(self):
        # Main update function
        self.update_longitudinal_physics()
        self.update_lateral_physics()
        self.update_position()
        
        # Basic position limiting to keep car on the "road"
        if self.position[1] < 0.5:
            self.position[1] = 0.5

# DQN Network
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42, hidden_size1=128, hidden_size2=64, hidden_size3=None):
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Network architecture (expanded for lane following)
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        
        # Optional third hidden layer for more complex behavior
        if hidden_size3:
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.fc4 = nn.Linear(hidden_size3, action_size)
            self.has_third_layer = True
        else:
            self.fc3 = nn.Linear(hidden_size2, action_size)
            self.has_third_layer = False

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier initialization
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        if self.has_third_layer:
            x = torch.relu(self.fc3(x))
            return self.fc4(x)
        else:
            return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed=42):
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size

        # Default configuration
        self.config = {
            'gamma': 0.99,             # Discount factor
            'tau': 0.01,               # Soft update parameter
            'lr': 0.0005,              # Learning rate
            'buffer_size': 100000,     # Replay buffer size
            'batch_size': 64,          # Batch size
            'update_every': 4,         # How often to update the network
            'epsilon_start': 1.0,      # Starting epsilon for exploration
            'epsilon_end': 0.05,       # Minimum epsilon
            'epsilon_decay': 0.995,    # Decay factor
            'hidden_size1': 128,       # First hidden layer size
            'hidden_size2': 64,        # Second hidden layer size
            'hidden_size3': 32,        # Third hidden layer size (for more complex behavior)
        }

        # Override with provided config
        if config:
            self.config.update(config)

        # Q-Networks
        self.qnetwork_local = DQNetwork(
            state_size, 
            action_size, 
            hidden_size1=self.config['hidden_size1'],
            hidden_size2=self.config['hidden_size2'],
            hidden_size3=self.config['hidden_size3']
        ).to(device)
        
        self.qnetwork_target = DQNetwork(
            state_size, 
            action_size,
            hidden_size1=self.config['hidden_size1'],
            hidden_size2=self.config['hidden_size2'],
            hidden_size3=self.config['hidden_size3']
        ).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config['lr'])

        # Replay buffer
        self.memory = ReplayBuffer(self.config['buffer_size'], self.config['batch_size'])

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        self.epsilon = self.config['epsilon_start']

    def step(self, state, action, reward, next_state, done):
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.config['update_every']
        if self.t_step == 0 and len(self.memory) > self.config['batch_size']:
            experiences = self.memory.sample()
            self.learn(experiences, self.config['gamma'])

    def act(self, state, eps=None):
        if eps is None:
            eps = self.epsilon

        # Convert state to tensor for neural network
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Epsilon-greedy action selection
        if random.random() > eps:
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = nn.functional.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config['tau'])

        # Update epsilon
        self.epsilon = max(self.config['epsilon_end'], self.epsilon * self.config['epsilon_decay'])

        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filename):
        torch.save({
            'qnetwork_state_dict': self.qnetwork_local.state_dict(),
            'target_network_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.epsilon = checkpoint['epsilon']

# Environment wrapper to match the DQN format
class CarEnv:
    def __init__(self, target_speed=50.0/3.6, max_steps=1000):
        self.car = CarSimulation()
        self.target_speed = target_speed  # m/s
        self.max_steps = max_steps
        self.current_step = 0
        self.last_action = 3  # COAST_NEUTRAL
        
        # Lane parameters
        self.lane_width = LANE_PARAMS['lane_width']
        self.max_lateral_deviation = LANE_PARAMS['max_lateral_deviation']
        self.lateral_deviation_penalty = LANE_PARAMS['lateral_deviation_penalty']

        # Define action space to match our expanded C++ implementation
        # 27 actions: 9 throttle/brake actions x 3 steering options (left, neutral, right)
        self.actions = {
            # Neutral steering actions (0-8)
            0: (0.0, 1.0, 0.0),    # STRONG_BRAKE_NEUTRAL
            1: (0.0, 0.66, 0.0),   # MEDIUM_BRAKE_NEUTRAL
            2: (0.0, 0.33, 0.0),   # LIGHT_BRAKE_NEUTRAL
            3: (0.0, 0.0, 0.0),    # COAST_NEUTRAL
            4: (0.25, 0.0, 0.0),   # LIGHT_THROTTLE_NEUTRAL
            5: (0.5, 0.0, 0.0),    # MEDIUM_THROTTLE_NEUTRAL
            6: (0.75, 0.0, 0.0),   # STRONG_THROTTLE_NEUTRAL
            7: (1.0, 0.0, 0.0),    # FULL_THROTTLE_NEUTRAL
            8: None,               # NO_CHANGE_NEUTRAL
            
            # Left steering actions (9-17)
            9: (0.0, 1.0, 0.3 * CAR_PARAMS['max_steering_angle']),    # STRONG_BRAKE_LEFT
            10: (0.0, 0.66, 0.3 * CAR_PARAMS['max_steering_angle']),  # MEDIUM_BRAKE_LEFT
            11: (0.0, 0.33, 0.3 * CAR_PARAMS['max_steering_angle']),  # LIGHT_BRAKE_LEFT
            12: (0.0, 0.0, 0.3 * CAR_PARAMS['max_steering_angle']),   # COAST_LEFT
            13: (0.25, 0.0, 0.3 * CAR_PARAMS['max_steering_angle']),  # LIGHT_THROTTLE_LEFT
            14: (0.5, 0.0, 0.3 * CAR_PARAMS['max_steering_angle']),   # MEDIUM_THROTTLE_LEFT
            15: (0.75, 0.0, 0.3 * CAR_PARAMS['max_steering_angle']),  # STRONG_THROTTLE_LEFT
            16: (1.0, 0.0, 0.3 * CAR_PARAMS['max_steering_angle']),   # FULL_THROTTLE_LEFT
            17: None,                                             # NO_CHANGE_LEFT
            
            # Right steering actions (18-26)
            18: (0.0, 1.0, -0.3 * CAR_PARAMS['max_steering_angle']),    # STRONG_BRAKE_RIGHT
            19: (0.0, 0.66, -0.3 * CAR_PARAMS['max_steering_angle']),   # MEDIUM_BRAKE_RIGHT
            20: (0.0, 0.33, -0.3 * CAR_PARAMS['max_steering_angle']),   # LIGHT_BRAKE_RIGHT
            21: (0.0, 0.0, -0.3 * CAR_PARAMS['max_steering_angle']),    # COAST_RIGHT
            22: (0.25, 0.0, -0.3 * CAR_PARAMS['max_steering_angle']),   # LIGHT_THROTTLE_RIGHT
            23: (0.5, 0.0, -0.3 * CAR_PARAMS['max_steering_angle']),    # MEDIUM_THROTTLE_RIGHT
            24: (0.75, 0.0, -0.3 * CAR_PARAMS['max_steering_angle']),   # STRONG_THROTTLE_RIGHT
            25: (1.0, 0.0, -0.3 * CAR_PARAMS['max_steering_angle']),    # FULL_THROTTLE_RIGHT
            26: None,                                                # NO_CHANGE_RIGHT
        }

    def reset(self):
        self.car.reset(random_init=True)
        self.current_step = 0
        self.last_action = 3  # COAST_NEUTRAL (center action)
        return self._get_state()

    def step(self, action):
        self.current_step += 1

        # Apply action
        if action == 8 or action == 17 or action == 26:  # NO_CHANGE actions
            # Keep the previous controls
            pass
        else:
            throttle, brake, steering = self.actions[action]
            self.car.throttle = throttle
            self.car.brake = brake
            self.car.steering_angle = steering

        # Remember last action
        self.last_action = action

        # Update simulation
        self.car.update()

        # Get new state
        state = self._get_state()

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Also end if car goes too far off the lane
        if abs(self.car.lateral_deviation) > self.max_lateral_deviation:
            done = True
            reward -= 5.0  # Extra penalty for going off the road

        return state, reward, done, {}

    def _get_state(self):
        """Convert car state to input for neural network"""
        state = np.zeros(9)  # Expanded state vector

        # Original speed control state variables
        # Current speed (normalized)
        state[0] = self.car.speed / 40.0  # Assuming max speed around 40 m/s

        # Speed difference from target (normalized)
        state[1] = (self.car.speed - self.target_speed) / 40.0

        # Current acceleration (normalized)
        state[2] = self.car.acceleration[2] / 10.0  # Assuming max accel around 10 m/s²

        # Current throttle
        state[3] = self.car.throttle

        # Current brake
        state[4] = self.car.brake

        # Engine RPM (normalized)
        state[5] = self.car.engine_speed / 8000.0
        
        # New lane following state variables
        # Lateral deviation from lane center (normalized)
        state[6] = self.car.lateral_deviation / (self.lane_width / 2.0)  # Normalized to [-1, 1] for lane width
        
        # Heading error (normalized)
        state[7] = self.car.lane_heading_error / np.pi  # Normalized to [-1, 1]
        
        # Steering angle (normalized)
        state[8] = self.car.steering_angle / CAR_PARAMS['max_steering_angle']  # Normalized to [-1, 1]

        return state

    def _calculate_reward(self, action):
        # Combined reward for both speed control and lane following
        reward = 0.0
        
        # Speed control reward
        speed_diff_ratio = abs(self.car.speed - self.target_speed) / self.target_speed
        
        if speed_diff_ratio < 0.05:  # Within 5% of target
            reward += 0.2
        else:
            # Exponential reward based on relative error
            reward += 0.2 * np.exp(-speed_diff_ratio * 5.0)
        
        # Lane following reward
        # Normalize lateral deviation to [-1, 1] where lane edge is 1
        lateral_deviation_normalized = abs(self.car.lateral_deviation) / (self.lane_width / 2.0)
        
        # Higher reward for staying close to center
        if lateral_deviation_normalized < 0.2:  # Very close to center
            reward += 1.0
        else:
            # Penalty increases exponentially as the car moves away from the center
            reward -= self.lateral_deviation_penalty * lateral_deviation_normalized**2
        
        # Heading alignment reward
        heading_error_normalized = abs(self.car.lane_heading_error) / np.pi
        reward -= heading_error_normalized * 0.5  # Penalty for misalignment with lane
        
        # Control smoothness rewards
        if action != self.last_action and action not in [8, 17, 26] and self.last_action not in [8, 17, 26]:
            # Calculate the basic action type (throttle/brake)
            current_base_action = action % 9
            last_base_action = self.last_action % 9
            
            # Calculate the steering group
            current_steering_group = action // 9  # 0 = neutral, 1 = left, 2 = right
            last_steering_group = self.last_action // 9
            
            # Penalize large throttle/brake changes
            if abs(current_base_action - last_base_action) > 2:
                reward -= 0.2
            
            # Penalize steering direction changes
            if last_steering_group != current_steering_group:
                reward -= 0.1
        
        # Penalize extreme behavior
        if self.car.throttle > 0.8 and self.car.speed > self.target_speed * 1.1:
            reward -= 0.3
        
        if self.car.brake > 0.0 and self.car.speed < self.target_speed * 0.9:
            reward -= 0.3
        
        return reward

# Training function with enhanced metrics
def train_dqn(env, agent, n_episodes=1000, max_t=1000, target_score=100.0,
             print_every=10, save_every=100, save_dir="./models"):

    # Create directory for saving models if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    scores = []
    scores_window = deque(maxlen=100)

    # For plotting
    all_rewards = []
    all_avg_speeds = []
    all_avg_lateral_deviations = []
    all_epsilons = []

    # Time tracking
    start_time = time.time()
    best_score = -np.inf

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        total_speed = 0
        total_lateral_deviation = 0

        for t in range(max_t):
            # Select and perform an action
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # Store experience in replay memory and learn
            agent.step(state, action, reward, next_state, done)

            # Update state and score
            state = next_state
            score += reward
            total_speed += env.car.speed
            total_lateral_deviation += abs(env.car.lateral_deviation)

            if done:
                break

        # Save scores
        scores.append(score)
        scores_window.append(score)

        # Calculate average metrics for this episode
        avg_speed = (total_speed / (t+1)) * 3.6  # Convert to km/h
        avg_lateral_deviation = total_lateral_deviation / (t+1)

        # Save metrics for plotting
        all_rewards.append(score)
        all_avg_speeds.append(avg_speed)
        all_avg_lateral_deviations.append(avg_lateral_deviation)
        all_epsilons.append(agent.epsilon)

        # Print progress
        if i_episode % print_every == 0:
            mean_score = np.mean(scores_window)
            elapsed = time.time() - start_time
            print(f"Episode {i_episode}/{n_episodes} | "
                  f"Average Score: {mean_score:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Avg Speed: {avg_speed:.1f} km/h | "
                  f"Avg Lat Dev: {avg_lateral_deviation:.2f} m | "
                  f"Elapsed: {elapsed:.1f}s")

            # Plot progress
            clear_output(wait=True)
            plot_training_progress(all_rewards, all_avg_speeds, all_avg_lateral_deviations, all_epsilons,
                                   target_speed=env.target_speed*3.6)

            # Save model if we have a new best score
            if mean_score > best_score:
                best_score = mean_score
                agent.save(f"{save_dir}/best_lane_following_model.pth")
                print(f"New best model saved with score: {best_score:.2f}")

                # Export this model for C++
                export_model_for_cpp(f"{save_dir}/best_lane_following_model.pth", 
                                      f"{save_dir}/best_lane_following_model_for_cpp.txt")
                print(f"Best model exported for C++: {save_dir}/best_lane_following_model_for_cpp.txt")

        # Save checkpoint periodically
        if i_episode == 1 or i_episode % save_every == 0 or i_episode == n_episodes:
            checkpoint_path = f"{save_dir}/lane_following_checkpoint_{i_episode}.pth"
            agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # Also export this model for C++
            export_path = f"{save_dir}/lane_following_model_for_cpp_ep{i_episode}.txt"
            export_model_for_cpp(checkpoint_path, export_path)
            print(f"Model exported for C++: {export_path}")

        # Check if we've solved the environment
        if np.mean(scores_window) >= target_score and len(scores_window) >= 100:
            print(f"\nEnvironment solved in {i_episode} episodes! Average Score: {np.mean(scores_window):.2f}")
            agent.save(f"{save_dir}/lane_following_solved_model.pth")

            # Export this model for C++
            export_model_for_cpp(f"{save_dir}/lane_following_solved_model.pth", 
                                  f"{save_dir}/lane_following_solved_model_for_cpp.txt")
            print(f"Solved model exported for C++: {save_dir}/lane_following_solved_model_for_cpp.txt")
            break

    # Save final model
    agent.save(f"{save_dir}/lane_following_final_model.pth")
    export_model_for_cpp(f"{save_dir}/lane_following_final_model.pth", 
                          f"{save_dir}/lane_following_final_model_for_cpp.txt")
    
    # Save training stats
    np.save(f"{save_dir}/lane_following_training_stats.npy",
            {'rewards': all_rewards, 
             'speeds': all_avg_speeds, 
             'lateral_deviations': all_avg_lateral_deviations,
             'epsilons': all_epsilons})

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.1f} minutes")

    return scores

# Function to plot training progress with lane metrics
def plot_training_progress(rewards, speeds, lateral_deviations, epsilons, target_speed):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 15))

    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    # Plot moving average of rewards
    window_size = min(100, len(rewards))
    if window_size > 0:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(moving_avg, color='red')

    # Plot speeds
    ax2.plot(speeds)
    ax2.axhline(y=target_speed, color='r', linestyle='--', label=f'Target: {target_speed:.1f} km/h')
    ax2.set_title('Average Speed per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Speed (km/h)')
    ax2.legend()
    
    # Plot lateral deviations
    ax3.plot(lateral_deviations)
    ax3.axhline(y=0.0, color='g', linestyle='--', label='Lane Center')
    ax3.axhline(y=LANE_PARAMS['lane_width']/2, color='r', linestyle='--', label='Lane Edge')
    ax3.axhline(y=-LANE_PARAMS['lane_width']/2, color='r', linestyle='--')
    ax3.set_title('Average Lateral Deviation per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Deviation (m)')
    ax3.legend()

    # Plot epsilon
    ax4.plot(epsilons)
    ax4.set_title('Exploration Rate (Epsilon)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Epsilon')

    plt.tight_layout()
    plt.show()

# Function to export the model to a format usable by your C++ code
def export_model_for_cpp(model_path, output_path):
    # Load the PyTorch model
    checkpoint = torch.load(model_path)

    # Extract model parameters
    weights = []
    biases = []

    # Get config from checkpoint
    config = checkpoint.get('config', {})
    
    # Create a network with the same architecture
    hidden_size1 = config.get('hidden_size1', 128)
    hidden_size2 = config.get('hidden_size2', 64)
    hidden_size3 = config.get('hidden_size3', None)
    
    model = DQNetwork(9, 27, hidden_size1=hidden_size1, hidden_size2=hidden_size2, hidden_size3=hidden_size3)
    model.load_state_dict(checkpoint['qnetwork_state_dict'])

    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy())
        elif 'bias' in name:
            biases.append(param.data.cpu().numpy())

    # Export in a simple text format
    with open(output_path, 'w') as f:
        # Write network architecture
        f.write("network_architecture\n")

        # Input size
        f.write(f"{weights[0].shape[1]}\n")

        # Hidden layers and output layer sizes
        for w in weights:
            f.write(f"{w.shape[0]}\n")

        # Weights
        f.write("weights\n")
        for layer_weights in weights:
            for row in layer_weights:
                f.write(" ".join([str(w) for w in row]) + "\n")

        # Biases
        f.write("biases\n")
        for layer_biases in biases:
            f.write(" ".join([str(b) for b in layer_biases]) + "\n")

    print(f"Model exported to {output_path}")

# Curriculum learning stages for both speed and lane following
curriculum_stages = [
    {
        'name': 'Easy Speed Control',
        'speed_min_kmh': 30, 
        'speed_max_kmh': 40, 
        'target_score': 50,
        'lane_deviation_max': 0.5,  # Small initial deviation
        'lateral_deviation_penalty': 0.5  # Gentle penalty to focus on speed
    },
    {
        'name': 'Basic Lane Following',
        'speed_min_kmh': 30, 
        'speed_max_kmh': 50, 
        'target_score': 50,
        'lane_deviation_max': 1.0,  # Medium deviation
        'lateral_deviation_penalty': 1.0  # Standard penalty
    },
    {
        'name': 'Advanced Lane Control',
        'speed_min_kmh': 40, 
        'speed_max_kmh': 70, 
        'target_score': 50,
        'lane_deviation_max': 1.5,  # Larger deviation to recover from
        'lateral_deviation_penalty': 1.5  # Higher penalty for more strict lane keeping
    },
    {
        'name': 'Full Speed Range',
        'speed_min_kmh': 20, 
        'speed_max_kmh': 100, 
        'target_score': 45,
        'lane_deviation_max': 2.0,  # Full range
        'lateral_deviation_penalty': 1.0  # Standard penalty for final training
    }
]

def train_with_curriculum(env, agent, curriculum_stages, episodes_per_stage=500, 
                          max_t=1000, print_every=10, save_every=100, save_dir="./models"):
    """
    Train a DQN agent using curriculum learning with progressively more challenging 
    speed ranges and lane following tasks.
    """
    import numpy as np
    from collections import deque
    import time
    import os
    
    # Create directory for saving models if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Overall tracking
    all_scores = []
    all_rewards = []
    all_avg_speeds = []
    all_avg_lateral_deviations = []
    all_epsilons = []
    stage_results = []
    
    total_episodes = 0
    overall_start_time = time.time()
    
    print("Starting curriculum training with these stages:")
    for i, stage in enumerate(curriculum_stages):
        print(f"Stage {i+1}: {stage['name']} - Speed: {stage['speed_min_kmh']}-{stage['speed_max_kmh']} km/h, "
              f"Lane deviation max: {stage['lane_deviation_max']} m, Target score: {stage['target_score']}")
    
    # Train through each curriculum stage
    for stage_idx, stage in enumerate(curriculum_stages):
        print(f"\n{'='*50}")
        print(f"CURRICULUM STAGE {stage_idx+1}/{len(curriculum_stages)}: {stage['name']}")
        print(f"Speed Range: {stage['speed_min_kmh']}-{stage['speed_max_kmh']} km/h")
        print(f"Lane Deviation Max: {stage['lane_deviation_max']} m")
        print(f"Target Score: {stage['target_score']}")
        print(f"{'='*50}\n")
        
        # Set up tracking for this stage
        scores = []
        scores_window = deque(maxlen=100)
        stage_rewards = []
        stage_avg_speeds = []
        stage_avg_lateral_deviations = []
        stage_epsilons = []
        
        # Time tracking for this stage
        stage_start_time = time.time()
        
        # Convert km/h to m/s
        min_speed = stage['speed_min_kmh'] / 3.6
        max_speed = stage['speed_max_kmh'] / 3.6
        
        # Update environment configuration for this stage
        env.max_lateral_deviation = stage['lane_deviation_max']
        env.lateral_deviation_penalty = stage['lateral_deviation_penalty']
        
        for i_episode in range(1, episodes_per_stage+1):
            # Set a random target speed for this episode
            target_speed = np.random.uniform(min_speed, max_speed)
            env.target_speed = target_speed
            
            # Reset environment with the new target speed
            state = env.reset()
            score = 0
            total_speed = 0
            total_lateral_deviation = 0
            
            for t in range(max_t):
                # Select and perform action
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store experience and learn
                agent.step(state, action, reward, next_state, done)
                
                # Update state and tracking
                state = next_state
                score += reward
                total_speed += env.car.speed
                total_lateral_deviation += abs(env.car.lateral_deviation)
                
                if done:
                    break
            
            # Save scores and metrics
            scores.append(score)
            scores_window.append(score)
            all_scores.append(score)
            
            # Calculate average metrics for this episode
            avg_speed = (total_speed / (t+1)) * 3.6  # Convert to km/h
            avg_lateral_deviation = total_lateral_deviation / (t+1)
            
            # Save metrics for plotting
            stage_rewards.append(score)
            stage_avg_speeds.append(avg_speed)
            stage_avg_lateral_deviations.append(avg_lateral_deviation)
            stage_epsilons.append(agent.epsilon)
            
            all_rewards.append(score)
            all_avg_speeds.append(avg_speed)
            all_avg_lateral_deviations.append(avg_lateral_deviation)
            all_epsilons.append(agent.epsilon)
            
            total_episodes += 1
            
            # Print progress
            if i_episode % print_every == 0:
                mean_score = np.mean(scores_window)
                elapsed = time.time() - stage_start_time
                print(f"Stage {stage_idx+1}/{len(curriculum_stages)} | "
                      f"Episode {i_episode}/{episodes_per_stage} | "
                      f"Average Score: {mean_score:.2f} | "
                      f"Epsilon: {agent.epsilon:.4f} | "
                      f"Target: {target_speed*3.6:.1f} km/h | "
                      f"Avg Speed: {avg_speed:.1f} km/h | "
                      f"Avg Lat Dev: {avg_lateral_deviation:.2f} m | "
                      f"Elapsed: {elapsed:.1f}s")
                
                # Plot progress
                if hasattr(plt, 'ion'):  # Check if matplotlib is available in interactive mode
                    try:
                        plt.figure(figsize=(12, 15))
                        
                        # Plot rewards
                        plt.subplot(4, 1, 1)
                        plt.plot(stage_rewards)
                        plt.title(f'Stage {stage_idx+1}: {stage["name"]} - Rewards')
                        
                        # Plot speeds
                        plt.subplot(4, 1, 2)
                        plt.plot(stage_avg_speeds)
                        plt.axhline(y=min_speed*3.6, color='g', linestyle='--')
                        plt.axhline(y=max_speed*3.6, color='g', linestyle='--')
                        plt.title('Speed Performance')
                        
                        # Plot lateral deviations
                        plt.subplot(4, 1, 3)
                        plt.plot(stage_avg_lateral_deviations)
                        plt.axhline(y=0.0, color='g', linestyle='--')
                        plt.axhline(y=env.lane_width/2, color='r', linestyle='--')
                        plt.axhline(y=-env.lane_width/2, color='r', linestyle='--')
                        plt.title('Lateral Deviation')
                        
                        # Plot exploration rate
                        plt.subplot(4, 1, 4)
                        plt.plot(stage_epsilons)
                        plt.title('Exploration Rate')
                        
                        plt.tight_layout()
                        plt.show()
                    except Exception as e:
                        print(f"Error plotting: {e}")
            
            # Save checkpoint periodically
            if i_episode % save_every == 0 or i_episode == episodes_per_stage:
                stage_name = f"stage{stage_idx+1}_{stage['name'].replace(' ', '_')}"
                checkpoint_path = f"{save_dir}/lane_following_curriculum_{stage_name}_ep{i_episode}.pth"
                agent.save(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
                
                # Also export for C++
                export_path = f"{save_dir}/lane_following_curriculum_{stage_name}_ep{i_episode}_for_cpp.txt"
                export_model_for_cpp(checkpoint_path, export_path)
                print(f"Model exported for C++: {export_path}")
            
            # Check if we've achieved the target score for this stage
            if len(scores_window) >= 100 and np.mean(scores_window) >= stage['target_score']:
                print(f"\nStage {stage_idx+1} solved in {i_episode} episodes! "
                      f"Average Score: {np.mean(scores_window):.2f}")
                
                # Save the model for this stage
                stage_name = f"stage{stage_idx+1}_{stage['name'].replace(' ', '_')}"
                agent.save(f"{save_dir}/lane_following_curriculum_{stage_name}_solved.pth")
                
                # Export for C++
                export_model_for_cpp(
                    f"{save_dir}/lane_following_curriculum_{stage_name}_solved.pth",
                    f"{save_dir}/lane_following_curriculum_{stage_name}_solved_for_cpp.txt")
                break
        
        # Save stage results
        stage_results.append({
            'stage_idx': stage_idx,
            'name': stage['name'],
            'speed_range': (stage['speed_min_kmh'], stage['speed_max_kmh']),
            'lane_deviation_max': stage['lane_deviation_max'],
            'episodes': i_episode,
            'final_score': np.mean(scores_window),
            'target_achieved': np.mean(scores_window) >= stage['target_score']
        })
    
    # Training completed - save final model
    agent.save(f"{save_dir}/lane_following_curriculum_final_model.pth")
    export_model_for_cpp(
        f"{save_dir}/lane_following_curriculum_final_model.pth", 
        f"{save_dir}/lane_following_curriculum_final_model_for_cpp.txt")
    
    # Save curriculum training stats
    np.save(f"{save_dir}/lane_following_curriculum_training_stats.npy", {
        'rewards': all_rewards, 
        'speeds': all_avg_speeds, 
        'lateral_deviations': all_avg_lateral_deviations,
        'epsilons': all_epsilons,
        'stage_results': stage_results
    })
    
    total_time = time.time() - overall_start_time
    print(f"\nCurriculum training completed in {total_time/60:.1f} minutes")
    print(f"Total episodes: {total_episodes}")
    
    for idx, result in enumerate(stage_results):
        print(f"Stage {idx+1}: {result['name']} - "
              f"{result['speed_range'][0]}-{result['speed_range'][1]} km/h, "
              f"Lane Dev Max: {result['lane_deviation_max']}m - "
              f"{'✓' if result['target_achieved'] else '✗'} "
              f"Score: {result['final_score']:.2f} in {result['episodes']} episodes")
    
    return all_scores

if __name__ == "__main__":
    # Ensure model directory exists
    os.makedirs("./car_dqn_models", exist_ok=True)
    
    # Create environment and agent for lane following
    env = CarEnv(target_speed=0.0, max_steps=1000)  # Target speed will be set per episode
    
    agent = DQNAgent(
        state_size=9,  # Expanded state size for lane following
        action_size=27, # Expanded action space with steering
        config={
            'gamma': 0.99,
            'tau': 0.01,
            'lr': 0.0003,  # Slightly lower learning rate for more stable training
            'buffer_size': 200000,  # Larger buffer for more complex behavior
            'batch_size': 64,
            'update_every': 4,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.998,  # Slower decay for more exploration
            'hidden_size1': 128,  # Larger network
            'hidden_size2': 64,
            'hidden_size3': 32,  # Added third layer for more complex behavior
        }
    )
    
    # Train with curriculum learning
    scores = train_with_curriculum(
        env,
        agent,
        curriculum_stages=curriculum_stages,
        episodes_per_stage=500,
        max_t=1000,
        print_every=5,
        save_every=50,
        save_dir="./car_dqn_models"
    )
    
    print("Lane following curriculum training complete!")