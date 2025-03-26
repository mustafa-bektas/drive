# CarGame Lane Keeping DQN Training
# ===============================

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

# Car Physics Parameters
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

# Lane Keeping Environment Parameters
LANE_PARAMS = {
    'lane_width': 10.0,
    'max_lateral_deviation': 5.0,
}

# Car Simulation Class (simplified for lane keeping)
class CarSimulation:
    def __init__(self, time_step=1.0/60.0):
        self.position = np.array([0.0, 0.5, 0.0])  # x, y, z
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.speed = 0.0
        self.rotation = 0.0
        self.steering_angle = 0.0
        self.throttle = 0.5  # Fixed throttle for lane keeping training
        self.brake = 0.0
        self.time_step = time_step
        self.params = CAR_PARAMS
        self.lateral_velocity = 0.0
        self.yaw_rate = 0.0

    def reset(self, random_init=True):
        # Start with random lateral position and heading for lane keeping training
        self.position = np.array([0.0, 0.5, 0.0])
        if random_init:
            # Random lateral position within the lane
            self.position[0] = np.random.uniform(-LANE_PARAMS['lane_width']/3, LANE_PARAMS['lane_width']/3)
            # Random initial heading
            self.rotation = np.random.uniform(-0.2, 0.2)

        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.speed = 15.0  # Constant speed for lane keeping training
        self.steering_angle = 0.0
        self.throttle = 0.5
        self.brake = 0.0
        self.lateral_velocity = 0.0
        self.yaw_rate = 0.0

        # Initialize velocity based on speed and rotation
        self.velocity[0] = self.speed * np.sin(self.rotation)  # Lateral component
        self.velocity[2] = self.speed * np.cos(self.rotation)  # Longitudinal component

    def update(self):
        # Simple kinematic update focused on lateral movement

        # Update rotation based on steering angle
        # This is a simplified model for steering
        self.yaw_rate = self.speed * np.tan(self.steering_angle) / self.params['wheel_base']
        self.rotation += self.yaw_rate * self.time_step

        # Normalize rotation
        while self.rotation > 2 * np.pi:
            self.rotation -= 2 * np.pi
        while self.rotation < 0:
            self.rotation += 2 * np.pi

        # Compute longitudinal and lateral velocities
        self.velocity[0] = self.speed * np.sin(self.rotation)  # Lateral component
        self.velocity[2] = self.speed * np.cos(self.rotation)  # Longitudinal component

        # Update position
        self.position += self.velocity * self.time_step

        # Update lateral velocity for state calculation
        self.lateral_velocity = self.velocity[0]

# DQN Network
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Network architecture (similar to the speed control network)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

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
class LaneKeepingDQNAgent:
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

# Environment wrapper for Lane Keeping
class LaneKeepingEnv:
    def __init__(self, lane_width=LANE_PARAMS['lane_width'], max_steps=1000):
        self.car = CarSimulation()
        self.lane_width = lane_width
        self.max_lateral_deviation = LANE_PARAMS['max_lateral_deviation']
        self.max_steps = max_steps
        self.current_step = 0
        self.last_action = 3  # MAINTAIN_STEERING

        # Define action space
        self.action_size = 7
        self.steering_adjustments = {
            0: 0.05,    # TURN_HARD_LEFT
            1: 0.025,    # TURN_MEDIUM_LEFT
            2: 0.01,    # TURN_GENTLE_LEFT
            3: 0.0,     # MAINTAIN_STEERING
            4: -0.01,   # TURN_GENTLE_RIGHT
            5: -0.025,   # TURN_MEDIUM_RIGHT
            6: -0.05,   # TURN_HARD_RIGHT
        }

    def reset(self):
        self.car.reset(random_init=True)
        self.current_step = 0
        self.last_action = 3  # MAINTAIN_STEERING
        return self._get_state()

    def step(self, action):
        self.current_step += 1

        # Apply steering adjustment
        steering_adjustment = self.steering_adjustments[action]
        # Apply steering adjustment
        if action == 3:  # MAINTAIN_STEERING
            self.car.steering_angle *= 0.7  # Return to center at 30% rate
        else:
            self.car.steering_angle += steering_adjustment

        # Limit steering angle
        if self.car.steering_angle > self.car.params['max_steering_angle']:
            self.car.steering_angle = self.car.params['max_steering_angle']
        elif self.car.steering_angle < -self.car.params['max_steering_angle']:
            self.car.steering_angle = -self.car.params['max_steering_angle']

        # Update car simulation
        self.car.update()

        # Get new state
        state = self._get_state()

        # Calculate reward
        reward = self._calculate_reward(state, action)

        # Check if episode is done
        done = self._is_done()

        # Remember last action
        self.last_action = action

        return state, reward, done, {}

    def _get_state(self):
        """Convert car state to input for neural network"""
        state = np.zeros(5)

        # Lateral position from lane center (normalized by lane width)
        lateral_position = self.car.position[0]
        state[0] = lateral_position / (self.lane_width / 2.0)

        # Heading error (normalized)
        # Lane is along z-axis so heading error is just the rotation
        heading_error = self.car.rotation
        while heading_error > np.pi: heading_error -= 2.0 * np.pi
        while heading_error < -np.pi: heading_error += 2.0 * np.pi
        state[1] = heading_error / 1.0  # Normalized to typical range

        # Lateral velocity (normalized)
        state[2] = self.car.lateral_velocity / 5.0

        # Current steering angle (normalized)
        state[3] = self.car.steering_angle / self.car.params['max_steering_angle']

        # Distance to nearest lane boundary (normalized)
        distance_to_boundary = (self.lane_width / 2.0) - abs(lateral_position)
        state[4] = distance_to_boundary / (self.lane_width / 2.0)

        return state

    def _calculate_reward(self, state, action):
        reward = 0.0

        # Reward for staying in the center of the lane
        lateral_position = self.car.position[0]
        centering_reward = np.exp(-5.0 * abs(lateral_position))
        reward += centering_reward * 2.0

        # Reward for aligning with the lane direction
        heading_error = abs(state[1])
        alignment_reward = 1.0 - min(1.0, heading_error)
        reward += alignment_reward

        # Penalize abrupt steering changes
        if self.last_action != action and action != 3 and self.last_action != 3:
            action_diff = abs(action - self.last_action)
            if action_diff > 2:
                reward -= 0.5 * (action_diff - 2)

        # Penalize excessive steering angles
        steering_ratio = abs(self.car.steering_angle / self.car.params['max_steering_angle'])
        if steering_ratio > 0.8:
            reward -= 0.5 * (steering_ratio - 0.8) / 0.2

        # Strong penalty for going off the lane
        if abs(self.car.position[0]) > self.max_lateral_deviation:
            reward -= 10.0

        return reward

    def _is_done(self):
        # Check if episode is done
        if self.current_step >= self.max_steps:
            return True

        # Episode is done if car leaves the lane by too much
        if abs(self.car.position[0]) > self.max_lateral_deviation:
            return True

        return False

    def render(self):
        # Placeholder for rendering, not implemented
        pass

# Function to export the model to a format usable by C++ code
def export_model_for_cpp(model_path, output_path):
    # Load the PyTorch model
    checkpoint = torch.load(model_path)

    # Extract model parameters
    weights = []
    biases = []

    model = DQNetwork(5, 7)  # 5 state dimensions and 7 actions for lane keeping
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

# Training function
def train_lane_keeping_dqn(env, agent, n_episodes=1000, max_t=1000, target_score=90.0,
                         print_every=10, save_every=100, save_dir="./car_dqn_models"):

    # Create directory for saving models if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    scores = []
    scores_window = deque(maxlen=100)

    # For plotting
    all_rewards = []
    all_epsilons = []
    all_lateral_positions = []

    # Time tracking
    start_time = time.time()
    best_score = -np.inf

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        lateral_positions = []

        for t in range(max_t):
            # Select and perform an action
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # Store experience in replay memory and learn
            agent.step(state, action, reward, next_state, done)

            # Track lateral position
            lateral_positions.append(env.car.position[0])

            # Update state and score
            state = next_state
            score += reward

            if done:
                break

        # Save scores
        scores.append(score)
        scores_window.append(score)

        # Calculate average lateral position
        avg_lateral_position = np.mean([abs(pos) for pos in lateral_positions])

        # Save metrics for plotting
        all_rewards.append(score)
        all_epsilons.append(agent.epsilon)
        all_lateral_positions.append(avg_lateral_position)

        # Print progress
        if i_episode % print_every == 0:
            mean_score = np.mean(scores_window)
            elapsed = time.time() - start_time
            print(f"Episode {i_episode}/{n_episodes} | "
                  f"Average Score: {mean_score:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Avg Lateral Position: {avg_lateral_position:.2f} m | "
                  f"Elapsed: {elapsed:.1f}s")

            # Plot progress
            clear_output(wait=True)
            plot_training_progress(all_rewards, all_lateral_positions, all_epsilons)

            # Save model if we have a new best score
            if mean_score > best_score:
                best_score = mean_score
                agent.save(f"{save_dir}/best_lane_keeping_model.pth")
                print(f"New best model saved with score: {best_score:.2f}")

                # Export this model for C++
                export_model_for_cpp(f"{save_dir}/best_lane_keeping_model.pth",
                                   f"{save_dir}/best_lane_keeping_model_for_cpp.txt")
                print(f"Best model exported for C++: {save_dir}/best_lane_keeping_model_for_cpp.txt")

        # Save checkpoint periodically
        if i_episode == 1 or i_episode % save_every == 0 or i_episode == n_episodes:
            checkpoint_path = f"{save_dir}/lane_keeping_checkpoint_{i_episode}.pth"
            agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # Also export this model for C++
            export_path = f"{save_dir}/lane_keeping_model_for_cpp_ep{i_episode}.txt"
            export_model_for_cpp(checkpoint_path, export_path)
            print(f"Model exported for C++: {export_path}")

        # Check if we've solved the environment
        if np.mean(scores_window) >= target_score and len(scores_window) >= 100:
            print(f"\nEnvironment solved in {i_episode} episodes! Average Score: {np.mean(scores_window):.2f}")
            agent.save(f"{save_dir}/lane_keeping_solved_model.pth")

            # Export this model for C++
            export_model_for_cpp(f"{save_dir}/lane_keeping_solved_model.pth",
                               f"{save_dir}/lane_keeping_solved_model_for_cpp.txt")
            print(f"Solved model exported for C++: {save_dir}/lane_keeping_solved_model_for_cpp.txt")
            break

    # Save final model
    agent.save(f"{save_dir}/lane_keeping_final_model.pth")

    # Export final model for C++
    export_model_for_cpp(f"{save_dir}/lane_keeping_final_model.pth",
                       f"{save_dir}/lane_keeping_final_model_for_cpp.txt")
    print(f"Final model exported for C++: {save_dir}/lane_keeping_final_model_for_cpp.txt")

    # Save training stats
    np.save(f"{save_dir}/lane_keeping_training_stats.npy",
            {'rewards': all_rewards, 'lateral_positions': all_lateral_positions, 'epsilons': all_epsilons})

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.1f} minutes")

    return scores

# Function to plot training progress
def plot_training_progress(rewards, lateral_positions, epsilons):
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

    # Plot lateral positions
    ax2.plot(lateral_positions)
    ax2.axhline(y=0, color='r', linestyle='--', label='Lane Center')
    ax2.set_title('Average Lateral Position per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Lateral Position (m)')
    ax2.legend()

    # Plot epsilon
    ax3.plot(epsilons)
    ax3.set_title('Exploration Rate (Epsilon)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')

    plt.tight_layout()
    plt.show()

# Main execution block
if __name__ == "__main__":
    # Ensure model directory exists
    os.makedirs("./car_dqn_models", exist_ok=True)

    # Create environment and agent
    env = LaneKeepingEnv()

    agent = LaneKeepingDQNAgent(
        state_size=5,       # 5 state dimensions for lane keeping
        action_size=7,      # 7 discrete steering actions
        config={
            'gamma': 0.98,
            'tau': 0.01,
            'lr': 0.001,
            'buffer_size': 100000,
            'batch_size': 64,
            'update_every': 4,
            'epsilon_start': 1.5,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.997,
        }
    )

    # Train the agent
    scores = train_lane_keeping_dqn(
        env,
        agent,
        n_episodes=10000,
        max_t=1000,
        target_score=2500.0,
        print_every=10,
        save_every=50,
        save_dir="./car_dqn_models"
    )

    print("Lane keeping training complete!")