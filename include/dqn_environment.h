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
        float laneWidth;          // Width of the lane in meters
        float maxLateralDeviation; // Maximum allowed lateral deviation before episode ends
        float lateralDeviationPenalty; // Penalty factor for deviating from lane center
        
        // Constructor with default values
        Config() 
            : targetSpeed(50.0f / 3.6f),
              maxEpisodeSteps(1000),
              speedRewardThreshold(1.0f),
              timeStep(1.0f / 60.0f),
              actionSpace(27),    // Expanded for steering actions
              laneWidth(4.0f),
              maxLateralDeviation(2.5f),
              lateralDeviationPenalty(1.0f)
        {}
    };

    // Action space is discretized for DQN with combinations of throttle/brake and steering
    enum Action {
        // Original throttle/brake actions with neutral steering (index 0-8)
        STRONG_BRAKE_NEUTRAL = 0,
        MEDIUM_BRAKE_NEUTRAL = 1,
        LIGHT_BRAKE_NEUTRAL = 2,
        COAST_NEUTRAL = 3,
        LIGHT_THROTTLE_NEUTRAL = 4,
        MEDIUM_THROTTLE_NEUTRAL = 5,
        STRONG_THROTTLE_NEUTRAL = 6,
        FULL_THROTTLE_NEUTRAL = 7,
        NO_CHANGE_NEUTRAL = 8,
        
        // Left steering actions (index 9-17)
        STRONG_BRAKE_LEFT = 9,
        MEDIUM_BRAKE_LEFT = 10,
        LIGHT_BRAKE_LEFT = 11,
        COAST_LEFT = 12,
        LIGHT_THROTTLE_LEFT = 13,
        MEDIUM_THROTTLE_LEFT = 14,
        STRONG_THROTTLE_LEFT = 15,
        FULL_THROTTLE_LEFT = 16,
        NO_CHANGE_LEFT = 17,
        
        // Right steering actions (index 18-26)
        STRONG_BRAKE_RIGHT = 18,
        MEDIUM_BRAKE_RIGHT = 19,
        LIGHT_BRAKE_RIGHT = 20,
        COAST_RIGHT = 21,
        LIGHT_THROTTLE_RIGHT = 22,
        MEDIUM_THROTTLE_RIGHT = 23,
        STRONG_THROTTLE_RIGHT = 24,
        FULL_THROTTLE_RIGHT = 25,
        NO_CHANGE_RIGHT = 26
    };

    DQNEnvironment(Config config = Config());
    
    // Initialize or reset the environment
    std::vector<float> reset();
    
    // Step the simulation based on the agent's action
    std::tuple<std::vector<float>, float, bool> step(Action action);
    
    // Convert action enum to throttle/brake/steering values
    std::tuple<float, float, float> actionToControls(Action action);
    
    // Get the current car for visualization
    const Car& getCar() const { return car; }
    
    // Helper for visualization
    float getTargetSpeed() const { return config.targetSpeed; }
    int getStateSize() const { return 9; } // Expanded state vector size
    int getActionSize() const { return config.actionSpace; }
    
    // Get lane-related information
    float getLaneWidth() const { return config.laneWidth; }
    float getLateralDeviation() const { return lateralDeviation; }
    
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
    float currentSteering;
    
    // Lane following variables
    float lateralDeviation;   // Current deviation from lane center (negative = left, positive = right)
    float laneHeadingError;   // Heading error relative to lane direction
    
    std::vector<float> getState();
    float calculateReward(const std::vector<float>& state, Action action);
    void updateLaneInformation();
};

} // namespace CarGame