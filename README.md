# Autonomous Vehicle Control with Deep Reinforcement Learning

An autonomous vehicle control system using Deep Q-Networks (DQN) for speed regulation and lane keeping, made for SWE599 graduation project at Bogazici University.

## 🚗 Project Overview

This project implements a modular DQN architecture that separates longitudinal (speed) and lateral (lane keeping) control tasks. The system combines realistic vehicle dynamics with modern reinforcement learning techniques to achieve robust autonomous driving capabilities.

### Key Features

- **Modular DQN Architecture**: Separate neural networks for speed control and lane keeping
- **Realistic Vehicle Physics**: Comprehensive dynamics model with Pacejka tire model
- **Curriculum Learning**: Progressive training difficulty for stable learning
- **Real-time Visualization**: 3D rendering with live telemetry at 60 FPS
- **Cross-platform**: Python training pipeline with C++ inference engine

## 🏗️ System Architecture

```
Python Training (PyTorch) → Model Export → C++ Runtime
       ↓                                       ↓
   DQN Agents                          Real-time Control
 (Speed + Lane)                      (Physics + Visualization)
```

### Components

- **Training Pipeline**: Python/PyTorch with curriculum learning
- **Physics Engine**: Longitudinal and lateral vehicle dynamics
- **DQN Agents**: Independent neural networks for each control task
- **Visualization**: Raylib-based 3D rendering with telemetry

## 🧠 Neural Network Architecture

### Speed Control Agent
- **State Space (6D)**: Speed, speed error, acceleration, throttle, brake, RPM
- **Action Space (9)**: Brake levels, coast, throttle levels, no change
- **Network**: 6 → 128 → 64 → 9 (ReLU activation)

### Lane Keeping Agent
- **State Space (5D)**: Lateral position, heading error, lateral velocity, steering angle, distance to edge
- **Action Space (7)**: Steering adjustments (hard/medium/gentle left/right, maintain)
- **Network**: 5 → 128 → 64 → 7 (ReLU activation)

## 🏎️ Vehicle Dynamics

The system implements comprehensive vehicle physics including:

**Longitudinal Dynamics**:
- Engine torque production based on throttle and RPM
- Drivetrain simulation with gear ratios
- Pacejka tire model for slip ratio calculation
- Aerodynamic drag and rolling resistance

**Lateral Dynamics**:
- Bicycle model with front/rear slip angles
- Cornering forces using Pacejka's Magic Formula
- Yaw dynamics with inertial effects
- Load transfer effects on tire performance

### Pacejka Tire Model
```
F = D sin(C arctan(Bα - E(Bα - arctan(Bα))))
```

## 📚 Curriculum Learning

Training progresses through four stages:
1. **Stage 1**: 30-40 km/h (Easy)
2. **Stage 2**: 45-55 km/h (Medium)
3. **Stage 3**: 70-90 km/h (Hard)
4. **Stage 4**: 20-100 km/h (Full Range)

Each stage trains for 500 episodes with target score thresholds.

## 🚀 Getting Started

### Prerequisites

**Python Dependencies**:
```bash
pip install torch numpy matplotlib raylib-python
```

**C++ Dependencies**:
- CMake 3.14+
- Raylib 4.5.0
- C++14 compiler

### Building

```bash
# Clone repository
git clone https://github.com/yourusername/autonomous-vehicle-control.git
cd autonomous-vehicle-control

# Build C++ simulation
mkdir build && cd build
cmake ..
make

# Install Python dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train speed control model
cd pytorch
python dqn.py

# Train lane keeping model
python lane_keeping_dqn.py
```

### Running Simulation

```bash
# Run with default models
./CarGame

# Run with custom models
./CarGame --speed-model models/your_speed_model.txt --lane-model models/your_lane_model.txt
```

## 🎮 Controls

- **Space**: Pause/Resume simulation
- **R**: Reset environment
- **L**: Toggle lane keeping
- **Page Up/Down**: Adjust simulation speed
- **Tab**: Toggle UI
- **F**: Fullscreen
- **ESC**: Exit

## 📊 Results

### Performance Metrics
- **Speed Control**: ±5% accuracy within target speed
- **Lane Keeping**: ±0.5m center position maintenance
- **Real-time Performance**: 60 FPS with full physics simulation
- **Robustness**: Handles curved roads and varying speed conditions

### Training Convergence
The curriculum learning approach enables stable learning across all speed ranges, with each stage achieving target performance before progressing to increased difficulty.

## 📁 Project Structure

```
├── include/           # C++ header files
├── src/              # C++ source files
├── pytorch/          # Python training scripts
├── models/           # Trained model files
├── CMakeLists.txt    # Build configuration
└── README.md
```

### Key Files

- `src/trained_model_demo.cpp`: Main simulation program
- `pytorch/dqn.py`: Speed control training
- `pytorch/lane_keeping_dqn.py`: Lane keeping training
- `include/car.h`: Vehicle dynamics model
- `src/rendering.cpp`: 3D visualization system

## 🔬 Technical Details

### DQN Algorithm
- **Experience Replay**: 100K buffer size, batch size 64
- **Target Networks**: Updated every τ steps for stability
- **Exploration**: ε-greedy with decay (1.0 → 0.05)
- **Loss Function**: MSE between predicted and target Q-values

### Reward Engineering
- **Speed Control**: Exponential reward for target proximity
- **Lane Keeping**: Center alignment with smoothness penalties
- **Curriculum**: Progressive complexity enables robust learning

## 🎯 Future Work

- **Obstacle Avoidance**: Integration of dynamic obstacle detection
- **Complex Scenarios**: Intersections, traffic lights, multi-lane highways
- **Sensor Fusion**: LiDAR and camera data integration
- **Adversarial Testing**: Robustness evaluation under challenging conditions

## 📖 References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
- Bengio, Y., et al. (2009). Curriculum learning. *Proceedings of ICML*, 41-48.
- Pacejka, H.B. & Bakker, E. (1992). The magic formula tyre model. *Vehicle System Dynamics*, 21(sup001), 1-18.

## 👨‍💻 Author

**Mustafa Bektaş**  
Advisor: Emre Uğur  
Software Engineering Department

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
