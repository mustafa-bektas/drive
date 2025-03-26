# speed control dqn training
# todo: clean this up later

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

# make sure dir exists
os.makedirs("./car_dqn_models", exist_ok=True)

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")

# car params - match c++ impl
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

# car sim class
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
        # simple torque curve model
        return throttle * (400.0 + 250.0 * (rpm / 4000.0) * (1.0 - rpm / 8000.0))

    def calculate_slip_ratio(self, wheel_linear_speed, vehicle_speed):
        # slip ratio for traction
        speed_abs = abs(vehicle_speed)
        min_speed = 0.5

        if speed_abs < min_speed:
            result = (wheel_linear_speed - vehicle_speed) / min_speed
        else:
            result = (wheel_linear_speed - vehicle_speed) / speed_abs

        return np.clip(result, -1.0, 1.0)

    def calculate_tire_force(self, slip_ratio):
        # pacejka tire model
        D = 1.0
        C = 1.5
        B = 10.0
        E = 0.1
        Fz = 4000.0

        coefficient = D * np.sin(C * np.arctan(B * slip_ratio - E * (B * slip_ratio - np.arctan(B * slip_ratio))))
        return coefficient * Fz

    def get_resistance_forces(self):
        # drag + rolling resistance
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
        # longit vehicle dynamics
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

            # update pos/vel
            self.velocity[0] = self.speed * np.sin(self.rotation)
            self.velocity[2] = self.speed * np.cos(self.rotation)

            self.position[0] += self.velocity[0] * self.time_step
            self.position[2] += self.velocity[2] * self.time_step

    def update(self):
        # main update
        self.update_longitudinal_physics()

        # limit position to keep on "road"
        if self.position[1] < 0.5:
            self.position[1] = 0.5

# q-network
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # match c++ impl architecture
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # xavier init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# replay buffer
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

# dqn agent
class DQNAgent:
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size

        # defaults
        self.config = {
            'gamma': 0.99,             # discount
            'tau': 0.01,               # soft update
            'lr': 0.001,               # learning rate
            'buffer_size': 50000,      # buffer size
            'batch_size': 64,          # batch size
            'update_every': 4,         # update freq
            'epsilon_start': 1.0,      # start epsilon
            'epsilon_end': 0.05,       # min epsilon
            'epsilon_decay': 0.995,    # decay factor
        }

        # override with config
        if config:
            self.config.update(config)

        # q-nets
        self.qnetwork_local = DQNetwork(state_size, action_size).to(device)
        self.qnetwork_target = DQNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config['lr'])

        # replay buffer
        self.memory = ReplayBuffer(self.config['buffer_size'], self.config['batch_size'])

        # time step (for updating every update_every steps)
        self.t_step = 0
        self.epsilon = self.config['epsilon_start']

    def step(self, state, action, reward, next_state, done):
        # add to buffer
        self.memory.add(state, action, reward, next_state, done)

        # learn every update_every steps
        self.t_step = (self.t_step + 1) % self.config['update_every']
        if self.t_step == 0 and len(self.memory) > self.config['batch_size']:
            experiences = self.memory.sample()
            self.learn(experiences, self.config['gamma'])

    def act(self, state, eps=None):
        if eps is None:
            eps = self.epsilon

        # state -> tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # epsilon-greedy
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

        # max q-vals for next states from target
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # expected q-vals from local
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # loss
        loss = nn.functional.mse_loss(Q_expected, Q_targets)

        # minimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target net
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config['tau'])

        # update epsilon
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

# env wrapper for dqn format
class CarEnv:
    def __init__(self, target_speed=50.0/3.6, max_steps=1000):
        self.car = CarSimulation()
        self.target_speed = target_speed  # m/s
        self.max_steps = max_steps
        self.current_step = 0
        self.last_action = 3  # COAST

        # actions match c++ impl
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

        # apply action
        if action == 8:  # NO_CHANGE
            # keep prev controls
            pass
        else:
            throttle, brake = self.actions[action]
            self.car.throttle = throttle
            self.car.brake = brake

        # remember action
        self.last_action = action

        # update sim
        self.car.update()

        # get new state
        state = self._get_state()

        # reward
        reward = self._calculate_reward(action)

        # done?
        done = self.current_step >= self.max_steps

        return state, reward, done, {}

    def _get_state(self):
        """nn input state"""
        state = np.zeros(6)

        # curr speed (normalized)
        state[0] = self.car.speed / 40.0  # max ~40 m/s

        # speed diff from target (normalized)
        state[1] = (self.car.speed - self.target_speed) / 40.0

        # accel (normalized)
        state[2] = self.car.acceleration[0] / 10.0  # max ~10 m/s²

        # throttle
        state[3] = self.car.throttle

        # brake
        state[4] = self.car.brake

        # rpm (normalized)
        state[5] = self.car.engine_speed / 8000.0

        return state

    def _calculate_reward(self, action):
        # base reward
        reward = 0.0

        # rel error not abs diff
        speed_diff_ratio = abs(self.car.speed - self.target_speed) / self.target_speed
        
        if speed_diff_ratio < 0.05:  # within 5% of target
            reward += 0.2
        else:
            # exp reward based on rel error
            reward += 0.2 * np.exp(-speed_diff_ratio * 5.0)
        
        # smoothness reward (works across speeds)
        if action != self.last_action and action != 8 and self.last_action != 8:
            if abs(action - self.last_action) > 2:
                reward -= 0.2
        
        # penalize extreme behavior vs target
        if self.car.throttle > 0.8 and self.car.speed > self.target_speed * 1.1:
            reward -= 0.3
        
        if self.car.brake > 0.0 and self.car.speed < self.target_speed * 0.9:
            reward -= 0.3
        
        return reward

# training function
def train_dqn(env, agent, n_episodes=1000, max_t=1000, target_score=100.0,
             print_every=10, save_every=100, save_dir="./models"):

    # create dir if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    scores = []
    scores_window = deque(maxlen=100)

    # plot stuff
    all_rewards = []
    all_avg_speeds = []
    all_epsilons = []

    # timing
    start_time = time.time()
    best_score = -np.inf

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        total_speed = 0

        for t in range(max_t):
            # action
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # store exp & learn
            agent.step(state, action, reward, next_state, done)

            # update state/score
            state = next_state
            score += reward
            total_speed += env.car.speed

            if done:
                break

        # save scores
        scores.append(score)
        scores_window.append(score)

        # avg speed this ep
        avg_speed = (total_speed / (t+1)) * 3.6  # to km/h

        # save plot metrics
        all_rewards.append(score)
        all_avg_speeds.append(avg_speed)
        all_epsilons.append(agent.epsilon)

        # print progress
        if i_episode % print_every == 0:
            mean_score = np.mean(scores_window)
            elapsed = time.time() - start_time
            print(f"ep {i_episode}/{n_episodes} | "
                f"avg score: {mean_score:.2f} | "
                f"eps: {agent.epsilon:.4f} | "
                f"speed: {avg_speed:.1f} km/h | "
                f"time: {elapsed:.1f}s")

            # plot progress
            clear_output(wait=True)
            plot_training_progress(all_rewards, all_avg_speeds, all_epsilons,
                                   target_speed=env.target_speed*3.6)

            # save if new best
            if mean_score > best_score:
                best_score = mean_score
                agent.save(f"{save_dir}/best_model.pth")
                print(f"new best: {best_score:.2f}")

                # export for c++
                export_model_for_cpp(f"{save_dir}/best_model.pth", f"{save_dir}/best_model_for_cpp.txt")
                print(f"exported: {save_dir}/best_model_for_cpp.txt")

        # save checkpoint regularly
        if i_episode == 1 or i_episode % save_every == 0 or i_episode == n_episodes:
            checkpoint_path = f"{save_dir}/checkpoint_{i_episode}.pth"
            agent.save(checkpoint_path)
            print(f"saved: {checkpoint_path}")

            # export for c++
            export_path = f"{save_dir}/model_for_cpp_ep{i_episode}.txt"
            export_model_for_cpp(checkpoint_path, export_path)
            print(f"exported: {export_path}")

        # solved?
        if np.mean(scores_window) >= target_score and len(scores_window) >= 100:
            print(f"solved in {i_episode} eps. avg: {np.mean(scores_window):.2f}")
            agent.save(f"{save_dir}/solved_model.pth")

            # export for c++
            export_model_for_cpp(f"{save_dir}/solved_model.pth", f"{save_dir}/solved_model_for_cpp.txt")
            print(f"exported solved: {save_dir}/solved_model_for_cpp.txt")
            break

    # save final model
    agent.save(f"{save_dir}/final_model.pth")

    # export final c++ model
    export_model_for_cpp(f"{save_dir}/final_model.pth", f"{save_dir}/final_model_for_cpp.txt")
    print(f"exported final: {save_dir}/final_model_for_cpp.txt")

    # save training stats
    np.save(f"{save_dir}/final_training_stats.npy",
            {'rewards': all_rewards, 'speeds': all_avg_speeds, 'epsilons': all_epsilons})

    total_time = time.time() - start_time
    print(f"done in {total_time/60:.1f} min")

    return scores

# plot training progress
def plot_training_progress(rewards, speeds, epsilons, target_speed):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    # moving avg
    window_size = min(100, len(rewards))
    if window_size > 0:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(moving_avg, color='red')

    # speeds
    ax2.plot(speeds)
    ax2.axhline(y=target_speed, color='r', linestyle='--', label=f'Target: {target_speed:.1f} km/h')
    ax2.set_title('Average Speed per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Speed (km/h)')
    ax2.legend()

    # epsilon
    ax3.plot(epsilons)
    ax3.set_title('Exploration Rate (Epsilon)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')

    plt.tight_layout()
    plt.show()

# export for c++ use
def export_model_for_cpp(model_path, output_path):
    # load pytorch model
    checkpoint = torch.load(model_path)

    # extract params
    weights = []
    biases = []

    model = DQNetwork(6, 9)  # 6 states, 9 actions
    model.load_state_dict(checkpoint['qnetwork_state_dict'])

    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy())
        elif 'bias' in name:
            biases.append(param.data.cpu().numpy())

    # export in simple text format
    with open(output_path, 'w') as f:
        # network architecture
        f.write("network_architecture\n")

        # input size
        f.write(f"{weights[0].shape[1]}\n")

        # hidden/output layer sizes
        for w in weights:
            f.write(f"{w.shape[0]}\n")

        # weights
        f.write("weights\n")
        for layer_weights in weights:
            for row in layer_weights:
                f.write(" ".join([str(w) for w in row]) + "\n")

        # biases
        f.write("biases\n")
        for layer_biases in biases:
            f.write(" ".join([str(b) for b in layer_biases]) + "\n")

    print(f"exported to {output_path}")


def train_with_curriculum(env, agent, curriculum_stages, episodes_per_stage=500, 
                          max_t=1000, print_every=10, save_every=100, save_dir="./models"):
    """
    curriculum learning with progressively harder speed ranges
    """
    import numpy as np
    from collections import deque
    import time
    import os
    import pickle
    
    # create dir if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # tracking
    all_scores = []
    all_rewards = []
    all_avg_speeds = []
    all_epsilons = []
    stage_results = []
    
    total_episodes = 0
    overall_start_time = time.time()
    
    print("starting curriculum learning with stages::")
    for i, stage in enumerate(curriculum_stages):
        print(f"Stage {i+1}: {stage['speed_min_kmh']}-{stage['speed_max_kmh']} km/h, Target score: {stage['target_score']}")
    
    # train each stage
    for stage_idx, stage in enumerate(curriculum_stages):
        print(f"\n{'='*50}")
        print(f"CURRICULUM STAGE {stage_idx+1}/{len(curriculum_stages)}")
        print(f"Speed Range: {stage['speed_min_kmh']}-{stage['speed_max_kmh']} km/h")
        print(f"Target Score: {stage['target_score']}")
        print(f"{'='*50}\n")
        
        # stage tracking
        scores = []
        scores_window = deque(maxlen=100)
        stage_rewards = []
        stage_avg_speeds = []
        stage_epsilons = []
        
        # time
        stage_start_time = time.time()
        
        # kmh -> m/s
        min_speed = stage['speed_min_kmh'] / 3.6
        max_speed = stage['speed_max_kmh'] / 3.6
        
        for i_episode in range(1, episodes_per_stage+1):
            # random target this episode
            target_speed = np.random.uniform(min_speed, max_speed)
            env.target_speed = target_speed
            
            # reset env with new target
            state = env.reset()
            score = 0
            total_speed = 0
            
            for t in range(max_t):
                # action
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                # store & learn
                agent.step(state, action, reward, next_state, done)
                
                # update
                state = next_state
                score += reward
                total_speed += env.car.speed
                
                if done:
                    break
            
            # save scores/metrics
            scores.append(score)
            scores_window.append(score)
            all_scores.append(score)
            
            # avg speed this ep
            avg_speed = (total_speed / (t+1)) * 3.6  # to km/h
            
            # save metrics
            stage_rewards.append(score)
            stage_avg_speeds.append(avg_speed)
            stage_epsilons.append(agent.epsilon)
            
            all_rewards.append(score)
            all_avg_speeds.append(avg_speed)
            all_epsilons.append(agent.epsilon)
            
            total_episodes += 1
            
            # progress
            if i_episode % print_every == 0:
                mean_score = np.mean(scores_window)
                elapsed = time.time() - stage_start_time
                print(f"Stage {stage_idx+1}/{len(curriculum_stages)} | "
                      f"Episode {i_episode}/{episodes_per_stage} | "
                      f"Average Score: {mean_score:.2f} | "
                      f"Epsilon: {agent.epsilon:.4f} | "
                      f"Target: {target_speed*3.6:.1f} km/h | "
                      f"Avg Speed: {avg_speed:.1f} km/h | "
                      f"Elapsed: {elapsed:.1f}s")
                
                # plot progress
                if hasattr(plt, 'ion'):  # interactive mode check
                    try:
                        plt.figure(figsize=(12, 10))
                        plt.subplot(3, 1, 1)
                        plt.plot(stage_rewards)
                        plt.title(f'Stage {stage_idx+1} Rewards')
                        plt.subplot(3, 1, 2)
                        plt.plot(stage_avg_speeds)
                        plt.axhline(y=min_speed*3.6, color='g', linestyle='--')
                        plt.axhline(y=max_speed*3.6, color='g', linestyle='--')
                        plt.title('Speed Performance')
                        plt.subplot(3, 1, 3)
                        plt.plot(stage_epsilons)
                        plt.title('Exploration Rate')
                        plt.tight_layout()
                        plt.show()
                    except Exception as e:
                        print(f"Error plotting: {e}")
            
            # checkpoint
            if i_episode % save_every == 0 or i_episode == episodes_per_stage:
                stage_name = f"stage{stage_idx+1}_{stage['speed_min_kmh']}-{stage['speed_max_kmh']}kmh"
                checkpoint_path = f"{save_dir}/curriculum_{stage_name}_ep{i_episode}.pth"
                agent.save(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
                
                # export for c++
                export_path = f"{save_dir}/curriculum_{stage_name}_ep{i_episode}_for_cpp.txt"
                export_model_for_cpp(checkpoint_path, export_path)
                print(f"Model exported for C++: {export_path}")
            
            # target score reached?
            if len(scores_window) >= 100 and np.mean(scores_window) >= stage['target_score']:
                print(f"\nStage {stage_idx+1} solved in {i_episode} episodes! "
                      f"Average Score: {np.mean(scores_window):.2f}")
                
                # save model
                stage_name = f"stage{stage_idx+1}_{stage['speed_min_kmh']}-{stage['speed_max_kmh']}kmh"
                agent.save(f"{save_dir}/curriculum_{stage_name}_solved.pth")
                
                # export for c++
                export_model_for_cpp(
                    f"{save_dir}/curriculum_{stage_name}_solved.pth",
                    f"{save_dir}/curriculum_{stage_name}_solved_for_cpp.txt")
                break
        
        # save stage results
        stage_results.append({
            'stage_idx': stage_idx,
            'speed_range': (stage['speed_min_kmh'], stage['speed_max_kmh']),
            'episodes': i_episode,
            'final_score': np.mean(scores_window),
            'target_achieved': np.mean(scores_window) >= stage['target_score']
        })
    
    # save final model
    agent.save(f"{save_dir}/curriculum_final_model.pth")
    export_model_for_cpp(
        f"{save_dir}/curriculum_final_model.pth", 
        f"{save_dir}/curriculum_final_model_for_cpp.txt")
    
    # save curriculum stats
    np.save(f"{save_dir}/curriculum_training_stats.npy", {
        'rewards': all_rewards, 
        'speeds': all_avg_speeds, 
        'epsilons': all_epsilons,
        'stage_results': stage_results
    })
    
    total_time = time.time() - overall_start_time
    print(f"\nCurriculum training completed in {total_time/60:.1f} minutes")
    print(f"Total episodes: {total_episodes}")
    
    for idx, result in enumerate(stage_results):
        print(f"Stage {idx+1}: "
              f"{result['speed_range'][0]}-{result['speed_range'][1]} km/h - "
              f"{'✓' if result['target_achieved'] else '✗'} "
              f"Score: {result['final_score']:.2f} in {result['episodes']} episodes")
    
    return all_scores

# curriculum stages - adjust as needed
curriculum_stages = [
    {'speed_min_kmh': 30, 'speed_max_kmh': 40, 'target_score': 50},  # easy
    {'speed_min_kmh': 45, 'speed_max_kmh': 55, 'target_score': 50},  # medium
    {'speed_min_kmh': 70, 'speed_max_kmh': 90, 'target_score': 50},  # harder
    {'speed_min_kmh': 20, 'speed_max_kmh': 100, 'target_score': 45}  # full range
]


""" # Main execution - can run in colab
if __name__ == "__main__":
    # ensure models dir exists
    os.makedirs("./car_dqn_models", exist_ok=True)

    # target speed setup (m/s)
    target_speed_kmh = 50.0  # km/h
    target_speed = target_speed_kmh / 3.6  # to m/s

    env = CarEnv(target_speed=target_speed, max_steps=1000)

    # dqn agent
    agent = DQNAgent(
        state_size=6,  # 6 state dims matching c++
        action_size=9, # 9 discrete actions
        config={
            'gamma': 0.99,             # discount
            'tau': 0.01,               # soft update
            'lr': 0.001,               # learning rate
            'buffer_size': 100000,     # larger buffer
            'batch_size': 64,          # mini-batch size
            'update_every': 4,         # update freq
            'epsilon_start': 1.0,      # start explore
            'epsilon_end': 0.05,       # min explore
            'epsilon_decay': 0.995,    # decay rate
        }
    )

    # train
    scores = train_dqn(
        env,
        agent,
        n_episodes=2000,              # more episodes
        max_t=1000,                   # max steps per ep
        target_score=50.0,            # solved threshold
        print_every=5,                # print freq
        save_every=50,                # save freq 
        save_dir="./car_dqn_models"   # save dir
    )

    print("Training completed! Models exported for C++ use.") """

if __name__ == "__main__":
    # make sure dir exists
    os.makedirs("./car_dqn_models", exist_ok=True)
    
    # setup env and agent
    env = CarEnv(target_speed=0.0, max_steps=1000)  # target set per ep
    
    agent = DQNAgent(
        state_size=6,
        action_size=9,
        config={
            'gamma': 0.99,
            'tau': 0.01,
            'lr': 0.001,
            'buffer_size': 100000,
            'batch_size': 64,
            'update_every': 4,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995,
        }
    )
    
    # curriculum training
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
    
    print("Curriculum training complete")