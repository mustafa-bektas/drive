#pragma once

#include "car.h"
#include <vector>
#include <random>
#include <tuple>
#include <memory>

namespace CarGame {

class LaneKeepingEnvironment {
public:
    // Configuration for the lane keeping RL environment
    struct Config {
        float laneWidth;               // Width of the lane in meters
        float maxLateralDeviation;     // Maximum allowed lateral deviation before reset
        float maxEpisodeSteps;         // Maximum steps per episode
        float timeStep;                // Simulation time step
        int actionSpace;               // Number of discrete actions
        
        // Constructor with default values
        Config() 
            : laneWidth(10.0f),
              maxLateralDeviation(5.0f),
              maxEpisodeSteps(1000),
              timeStep(1.0f / 60.0f),
              actionSpace(7)
        {}
    };

    // Action space is discretized for DQN
    enum Action {
        TURN_HARD_LEFT = 0,     // Large left steering angle change
        TURN_MEDIUM_LEFT = 1,   // Medium left steering angle change
        TURN_GENTLE_LEFT = 2,   // Small left steering angle change
        MAINTAIN_STEERING = 3,  // No change to steering
        TURN_GENTLE_RIGHT = 4,  // Small right steering angle change
        TURN_MEDIUM_RIGHT = 5,  // Medium right steering angle change
        TURN_HARD_RIGHT = 6     // Large right steering angle change
    };

    LaneKeepingEnvironment(Config config = Config());
    
    // Initialize or reset the environment
    std::vector<float> reset(const Car& carRef);
    
    // Step the simulation based on the agent's action
    std::tuple<std::vector<float>, float, bool> step(Action action, Car& car);
    
    // Convert action enum to steering angle adjustment
    float actionToSteeringAdjustment(Action action);
    
    // Helper for visualization
    int getStateSize() const { return 5; } // State vector size
    int getActionSize() const { return config.actionSpace; }
    
    // Get lane position information
    float getLateralPosition(const Car& car) const;
    float getHeadingError(const Car& car) const;
    float getLateralVelocity(const Car& car) const;
    
private:
    Config config;
    int currentStep;
    std::mt19937 rng;
    
    // Last action for smoothness calculation
    Action lastAction;
    
    // Current lateral state
    float lateralPosition; // Distance from lane center
    float headingError;    // Difference between car heading and lane direction
    float lateralVelocity; // Rate of change of lateral position
    
    std::vector<float> getState(const Car& car);
    float calculateReward(const std::vector<float>& state, Action action, const Car& car);
};

} // namespace CarGame