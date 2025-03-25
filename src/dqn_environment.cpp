#include "dqn_environment.h"
#include <algorithm>
#include <cmath>

namespace CarGame {

DQNEnvironment::DQNEnvironment(Config config) 
    : config(config), 
      car(Vector3{0.0f, 0.5f, 0.0f}),
      currentStep(0),
      rng(std::random_device{}()),
      lastAction(COAST),
      currentThrottle(0.0f),
      currentBrake(0.0f) {
}

std::vector<float> DQNEnvironment::reset() {
    // Reset to a random position within the lane
    float laneWidth = 10.0f; // Match the lane width defined in rendering.cpp
    float randomLateralPosition = ((float)rand() / RAND_MAX - 0.5f) * laneWidth * 0.8f;
    
    // Random initial rotation (slight heading variation)
    float randomRotation = ((float)rand() / RAND_MAX - 0.5f) * 0.2f; // ±0.1 radians
    
    car = Car(Vector3{randomLateralPosition, 0.5f, 0.0f});
    car.rotation = randomRotation;
    currentStep = 0;
    
    // Reset control values for NO_CHANGE action
    currentThrottle = 0.0f;
    currentBrake = 0.0f;
    lastAction = COAST;
    
    return getState();
}

std::tuple<std::vector<float>, float, bool> DQNEnvironment::step(Action action) {
    // Apply the agent's action to the car
    std::pair<float, float> controls = actionToControls(action);
    car.throttle = controls.first;
    car.brake = controls.second;
    
    // Update current controls for NO_CHANGE action
    currentThrottle = controls.first;
    currentBrake = controls.second;
    
    // Update car physics
    car.update(config.timeStep);
    currentStep++;
    
    // Get the new state
    std::vector<float> newState = getState();
    
    // Calculate reward
    float reward = calculateReward(newState, action);
    
    // Check if episode is done
    bool done = currentStep >= config.maxEpisodeSteps;
    
    // Store last action for next reward calculation
    lastAction = action;
    
    return std::make_tuple(newState, reward, done);
}

std::pair<float, float> DQNEnvironment::actionToControls(Action action) {
    float throttle = 0.0f;
    float brake = 0.0f;
    
    switch(action) {
        case STRONG_BRAKE:
            brake = 1.0f;
            break;
        case MEDIUM_BRAKE:
            brake = 0.66f;
            break;
        case LIGHT_BRAKE:
            brake = 0.33f;
            break;
        case COAST:
            // Both zero
            break;
        case LIGHT_THROTTLE:
            throttle = 0.25f;
            break;
        case MEDIUM_THROTTLE:
            throttle = 0.5f;
            break;
        case STRONG_THROTTLE:
            throttle = 0.75f;
            break;
        case FULL_THROTTLE:
            throttle = 1.0f;
            break;
        case NO_CHANGE:
            throttle = currentThrottle;
            brake = currentBrake;
            break;
    }
    
    return std::make_pair(throttle, brake);
}

std::vector<float> DQNEnvironment::getState() {
    std::vector<float> state(6);
    
    // Current speed (normalized)
    state[0] = car.speed / 40.0f;  // Assuming max speed around 40 m/s (144 km/h)
    
    // Speed difference from target (normalized)
    state[1] = (car.speed - config.targetSpeed) / 40.0f;
    
    // Current acceleration (normalized)
    state[2] = car.acceleration.x / 10.0f;  // Assuming max accel around 10 m/s²
    
    // Current throttle
    state[3] = car.throttle;
    
    // Current brake
    state[4] = car.brake;
    
    // Engine RPM (normalized)
    state[5] = car.engineSpeed / 8000.0f;  // Based on max RPM in car.cpp
    
    return state;
}

float DQNEnvironment::calculateReward(const std::vector<float>& state, Action action) {
    float reward = 0.0f;
    
    // Main reward: how close the car is to the target speed
    float speedDiff = std::abs(car.speed - config.targetSpeed);
    
    if (speedDiff < config.speedRewardThreshold) {
        // Maximum reward when within threshold
        reward += 1.0f;
    } else {
        // Gradually decreasing reward as the difference increases
        reward += std::exp(-speedDiff * 0.5f);
    }
    
    // Penalize large changes in controls
    if (lastAction != action && 
        action != NO_CHANGE && 
        lastAction != NO_CHANGE) {
        // Only penalize if action changed significantly
        if (std::abs(static_cast<int>(action) - static_cast<int>(lastAction)) > 2) {
            reward -= 0.2f;
        }
    }
    
    // Penalize extreme throttle changes
    if (car.throttle > 0.8f && car.speed > config.targetSpeed * 1.1f) {
        reward -= 0.3f;
    }
    
    // Penalize unnecessary braking
    if (car.brake > 0.0f && car.speed < config.targetSpeed * 0.9f) {
        reward -= 0.3f;
    }
    
    return reward;
}

} // namespace CarGame