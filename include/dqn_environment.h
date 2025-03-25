#pragma once

#include "car.h"
#include <vector>
#include <random>
#include <tuple>
#include <memory>

namespace CarGame {

class DQNEnvironment {
public:
    // Configuration for the RL environment
    struct Config {
        float targetSpeed;        // Target speed in m/s (50 km/h)
        float maxEpisodeSteps;    // Maximum steps per episode
        float speedRewardThreshold; // Speed difference threshold for max reward
        float timeStep;           // Simulation time step
        int actionSpace;          // Number of discrete actions
        
        // Constructor with default values
        Config() 
            : targetSpeed(50.0f / 3.6f),
              maxEpisodeSteps(1000),
              speedRewardThreshold(1.0f),
              timeStep(1.0f / 60.0f),
              actionSpace(9)
        {}
    };

    // Action space is discretized for DQN
    enum Action {
        STRONG_BRAKE = 0,    // Brake 1.0
        MEDIUM_BRAKE = 1,    // Brake 0.66
        LIGHT_BRAKE = 2,     // Brake 0.33
        COAST = 3,           // No throttle or brake
        LIGHT_THROTTLE = 4,  // Throttle 0.25
        MEDIUM_THROTTLE = 5, // Throttle 0.5
        STRONG_THROTTLE = 6, // Throttle 0.75
        FULL_THROTTLE = 7,   // Throttle 1.0
        NO_CHANGE = 8        // Keep current throttle/brake
    };

    DQNEnvironment(Config config = Config());
    
    // Initialize or reset the environment
    std::vector<float> reset();
    
    // Step the simulation based on the agent's action
    std::tuple<std::vector<float>, float, bool> step(Action action);
    
    // Convert action enum to throttle/brake values
    std::pair<float, float> actionToControls(Action action);
    
    // Get the current car for visualization
    const Car& getCar() const { return car; }

    // Get the current car for modification (non-const version)
    Car& getCar() { return car; }
    
    // Helper for visualization
    float getTargetSpeed() const { return config.targetSpeed; }
    int getStateSize() const { return 6; } // State vector size
    int getActionSize() const { return config.actionSpace; }
    
private:
    Car car;
    Config config;
    int currentStep;
    std::mt19937 rng;
    
    // Last action for smoothness calculation
    Action lastAction;
    
    // Current controls (for NO_CHANGE action)
    float currentThrottle;
    float currentBrake;
    
    std::vector<float> getState();
    float calculateReward(const std::vector<float>& state, Action action);
};

} // namespace CarGame