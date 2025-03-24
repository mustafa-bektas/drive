# CarGame DQN Training on Google Colab
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

    def reset(self, random_init=True):
        self.position = np.array([0.0, 0.5, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.speed = 0.0
        if random_init:
            self.speed = np.random.uniform(0, 5.0)
            self.velocity[0] = self.speed
        self.rotation = 0.0
        self.steering_angle = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.engine_speed = CAR_PARAMS['idle_rpm']
        self.engine_speed_dot = 0.0
        self.wheel_rotation_speed = 0.0
        self.clutch = True

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

            self.acceleration[0] = net_force / self.params['mass']
            self.speed += self.acceleration[0] * self.time_step

            # Update position and velocity
            self.velocity[0] = self.speed * np.sin(self.rotation)
            self.velocity[2] = self.speed * np.cos(self.rotation)

            self.position[0] += self.velocity[0] * self.time_step
            self.position[2] += self.velocity[2] * self.time_step

    def update(self):
        # Main update function
        self.update_longitudinal_physics()

        # Basic position limiting to keep car on the "road"
        if self.position[1] < 0.5:
            self.position[1] = 0.5

# DQN Network
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Network architecture (matching your C++ implementation)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

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
            'lr': 0.001,               # Learning rate
            'buffer_size': 50000,      # Replay buffer size
            'batch_size': 64,          # Batch size
            'update_every': 4,         # How often to update the network
            'epsilon_start': 1.0,      # Starting epsilon for exploration
            'epsilon_end': 0.05,       # Minimum epsilon
            'epsilon_decay': 0.995,    # Decay factor
        }

        # Override with provided config
        if config:
            self.config.update(config)

        # Q-Networks
        self.qnetwork_local = DQNetwork(state_size, action_size).to(device)
        self.qnetwork_target = DQNetwork(state_size, action_size).to(device)
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
        self.last_action = 3  # COAST

        # Define action space to match your C++ implementation
        self.actions = {
            0: (0.0, 1.0),    # STRONG_BRAKE
            1: (0.0, 0.66),   # MEDIUM_BRAKE
            2: (0.0, 0.33),   # LIGHT_BRAKE
            3: (0.0, 0.0),    # COAST
            4: (0.25, 0.0),   # LIGHT_THROTTLE
            5: (0.5, 0.0),    # MEDIUM_THROTTLE
            6: (0.75, 0.0),   # STRONG_THROTTLE
            7: (1.0, 0.0),    # FULL_THROTTLE
            8: None,          # NO_CHANGE - use last controls
        }

    def reset(self):
        self.car.reset(random_init=True)
        self.current_step = 0
        self.last_action = 3  # COAST
        return self._get_state()

    def step(self, action):
        self.current_step += 1

        # Apply action
        if action == 8:  # NO_CHANGE
            # Keep the previous controls
            pass
        else:
            throttle, brake = self.actions[action]
            self.car.throttle = throttle
            self.car.brake = brake

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

        return state, reward, done, {}

    def _get_state(self):
        """Convert car state to input for neural network"""
        state = np.zeros(6)

        # Current speed (normalized)
        state[0] = self.car.speed / 40.0  # Assuming max speed around 40 m/s

        # Speed difference from target (normalized)
        state[1] = (self.car.speed - self.target_speed) / 40.0

        # Current acceleration (normalized)
        state[2] = self.car.acceleration[0] / 10.0  # Assuming max accel around 10 m/s²

        # Current throttle
        state[3] = self.car.throttle

        # Current brake
        state[4] = self.car.brake

        # Engine RPM (normalized)
        state[5] = self.car.engine_speed / 8000.0

        return state

    def _calculate_reward(self, action):
        # FIXED: Adjusted reward function for better learning
        reward = 0.0

        # Main reward: how close the car is to the target speed
        speed_diff = abs(self.car.speed - self.target_speed)

        if speed_diff < 1.0:
            # Maximum reward when within threshold (reduced value)
            reward += 0.2  # Changed from 1.0
        else:
            # Gradually decreasing reward as the difference increases
            reward += 0.2 * np.exp(-speed_diff * 0.5)  # Scaled down

        # Penalize large changes in controls
        if action != self.last_action and action != 8 and self.last_action != 8:
            # Only penalize if action changed significantly
            if abs(action - self.last_action) > 2:
                reward -= 0.2

        # Penalize extreme throttle changes
        if self.car.throttle > 0.8 and self.car.speed > self.target_speed * 1.1:
            reward -= 0.3

        # Penalize unnecessary braking
        if self.car.brake > 0.0 and self.car.speed < self.target_speed * 0.9:
            reward -= 0.3

        return reward

# Training function
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
    all_epsilons = []

    # Time tracking
    start_time = time.time()
    best_score = -np.inf

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        total_speed = 0

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

            if done:
                break

        # Save scores
        scores.append(score)
        scores_window.append(score)

        # Calculate average speed for this episode
        avg_speed = (total_speed / (t+1)) * 3.6  # Convert to km/h

        # Save metrics for plotting
        all_rewards.append(score)
        all_avg_speeds.append(avg_speed)
        all_epsilons.append(agent.epsilon)

        # Print progress
        if i_episode % print_every == 0:
            mean_score = np.mean(scores_window)
            elapsed = time.time() - start_time
            print(f"Episode {i_episode}/{n_episodes} | "
                  f"Average Score: {mean_score:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Avg Speed: {avg_speed:.1f} km/h | "
                  f"Elapsed: {elapsed:.1f}s")

            # Plot progress
            clear_output(wait=True)
            plot_training_progress(all_rewards, all_avg_speeds, all_epsilons,
                                   target_speed=env.target_speed*3.6)

            # Save model if we have a new best score
            if mean_score > best_score:
                best_score = mean_score
                agent.save(f"{save_dir}/best_model.pth")
                print(f"New best model saved with score: {best_score:.2f}")

                # Export this model for C++
                export_model_for_cpp(f"{save_dir}/best_model.pth", f"{save_dir}/best_model_for_cpp.txt")
                print(f"Best model exported for C++: {save_dir}/best_model_for_cpp.txt")

        # FIXED: Save checkpoint periodically regardless of performance
        if i_episode == 1 or i_episode % save_every == 0 or i_episode == n_episodes:
            checkpoint_path = f"{save_dir}/checkpoint_{i_episode}.pth"
            agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # Also export this model for C++
            export_path = f"{save_dir}/model_for_cpp_ep{i_episode}.txt"
            export_model_for_cpp(checkpoint_path, export_path)
            print(f"Model exported for C++: {export_path}")

        # FIXED: Check if we've solved the environment - require enough episodes AND high score
        if np.mean(scores_window) >= target_score and len(scores_window) >= 100:
            print(f"\nEnvironment solved in {i_episode} episodes! Average Score: {np.mean(scores_window):.2f}")
            agent.save(f"{save_dir}/solved_model.pth")

            # Export this model for C++
            export_model_for_cpp(f"{save_dir}/solved_model.pth", f"{save_dir}/solved_model_for_cpp.txt")
            print(f"Solved model exported for C++: {save_dir}/solved_model_for_cpp.txt")
            break

    # Save final model
    agent.save(f"{save_dir}/final_model.pth")

    # Export final model for C++
    export_model_for_cpp(f"{save_dir}/final_model.pth", f"{save_dir}/final_model_for_cpp.txt")
    print(f"Final model exported for C++: {save_dir}/final_model_for_cpp.txt")

    # Save training stats
    np.save(f"{save_dir}/final_training_stats.npy",
            {'rewards': all_rewards, 'speeds': all_avg_speeds, 'epsilons': all_epsilons})

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.1f} minutes")

    return scores

# Function to plot training progress
def plot_training_progress(rewards, speeds, epsilons, target_speed):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

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

    # Plot epsilon
    ax3.plot(epsilons)
    ax3.set_title('Exploration Rate (Epsilon)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')

    plt.tight_layout()
    plt.show()

# Function to export the model to a format usable by your C++ code
def export_model_for_cpp(model_path, output_path):
    # Load the PyTorch model
    checkpoint = torch.load(model_path)

    # Extract model parameters
    weights = []
    biases = []

    model = DQNetwork(6, 9)  # Assuming 6 state dimensions and 9 actions
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

# Main execution - can be run directly in Colab
if __name__ == "__main__":
    # Ensure model directory exists
    os.makedirs("./car_dqn_models", exist_ok=True)

    # Set up the environment with specified target speed (in m/s)
    target_speed_kmh = 50.0  # km/h
    target_speed = target_speed_kmh / 3.6  # Convert to m/s

    env = CarEnv(target_speed=target_speed, max_steps=1000)

    # Create DQN agent
    agent = DQNAgent(
        state_size=6,  # 6 state dimensions matching your C++ implementation
        action_size=9, # 9 discrete actions
        config={
            'gamma': 0.99,             # Discount factor
            'tau': 0.01,               # Soft update parameter
            'lr': 0.001,               # Learning rate
            'buffer_size': 100000,     # Larger replay buffer
            'batch_size': 64,          # Mini-batch size
            'update_every': 4,         # How often to update the network
            'epsilon_start': 1.0,      # Starting exploration rate
            'epsilon_end': 0.05,       # Minimum exploration rate
            'epsilon_decay': 0.995,    # Exploration decay rate
        }
    )

    # Train the agent
    scores = train_dqn(
        env,
        agent,
        n_episodes=2000,              # More episodes for better training
        max_t=1000,                   # Max timesteps per episode
        target_score=50.0,            # Score threshold to consider solved
        print_every=5,                # Print more often
        save_every=50,                # Save more frequently
        save_dir="./car_dqn_models"   # Directory to save models
    )

    print("Training completed! The saved models have been exported for C++ use.")