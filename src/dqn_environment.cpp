#include "dqn_environment.h"
#include <algorithm>
#include <cmath>

namespace CarGame {

DQNEnvironment::DQNEnvironment(Config config) 
    : config(config), 
      car(Vector3{0.0f, 0.5f, 0.0f}),
      currentStep(0),
      rng(std::random_device{}()),
      lastAction(COAST_NEUTRAL),
      currentThrottle(0.0f),
      currentBrake(0.0f),
      currentSteering(0.0f),
      lateralDeviation(0.0f),
      laneHeadingError(0.0f) {
}

std::vector<float> DQNEnvironment::reset() {
    // Reset the car to initial state
    car = Car(Vector3{0.0f, 0.5f, 0.0f});
    currentStep = 0;
    
    // Reset control values
    currentThrottle = 0.0f;
    currentBrake = 0.0f;
    currentSteering = 0.0f;
    car.throttle = 0.0f;
    car.brake = 0.0f;
    car.steeringAngle = 0.0f;
    lastAction = COAST_NEUTRAL;
    
    // Reset lane variables
    lateralDeviation = 0.0f;
    laneHeadingError = 0.0f;
    
    // Add some randomization for better generalization
    std::uniform_real_distribution<float> speedDist(0.0f, 5.0f);
    std::uniform_real_distribution<float> lateralDist(-1.0f, 1.0f);
    
    float initialSpeed = speedDist(rng);
    car.speed = initialSpeed;
    car.velocity = {0.0f, 0.0f, initialSpeed};
    
    // Add some initial lateral deviation for training robustness
    lateralDeviation = lateralDist(rng);
    car.position.x = lateralDeviation;
    
    return getState();
}

std::tuple<std::vector<float>, float, bool> DQNEnvironment::step(Action action) {
    // Apply the agent's action to the car
    auto [throttle, brake, steering] = actionToControls(action);
    car.throttle = throttle;
    car.brake = brake;
    car.steeringAngle = steering;
    
    // Update current controls for NO_CHANGE action
    currentThrottle = throttle;
    currentBrake = brake;
    currentSteering = steering;
    
    // Update car physics
    car.update(config.timeStep);
    currentStep++;
    
    // Update lane information
    updateLaneInformation();
    
    // Get the new state
    std::vector<float> newState = getState();
    
    // Calculate reward
    float reward = calculateReward(newState, action);
    
    // Check if episode is done
    bool done = currentStep >= config.maxEpisodeSteps;
    
    // Also end if we go too far off the lane
    if (std::abs(lateralDeviation) > config.maxLateralDeviation) {
        done = true;
        reward -= 5.0f; // Extra penalty for going off the road
    }
    
    // Store last action for next reward calculation
    lastAction = action;
    
    return std::make_tuple(newState, reward, done);
}

std::tuple<float, float, float> DQNEnvironment::actionToControls(Action action) {
    float throttle = 0.0f;
    float brake = 0.0f;
    float steering = 0.0f;
    
    // First determine throttle/brake based on the action group
    int actionGroup = action / 9; // 0 = neutral, 1 = left, 2 = right
    int throttleBrakeAction = action % 9; // 0-8 correspond to original actions
    
    switch(throttleBrakeAction) {
        case 0: // STRONG_BRAKE
            brake = 1.0f;
            break;
        case 1: // MEDIUM_BRAKE
            brake = 0.66f;
            break;
        case 2: // LIGHT_BRAKE
            brake = 0.33f;
            break;
        case 3: // COAST
            // Both zero
            break;
        case 4: // LIGHT_THROTTLE
            throttle = 0.25f;
            break;
        case 5: // MEDIUM_THROTTLE
            throttle = 0.5f;
            break;
        case 6: // STRONG_THROTTLE
            throttle = 0.75f;
            break;
        case 7: // FULL_THROTTLE
            throttle = 1.0f;
            break;
        case 8: // NO_CHANGE
            throttle = currentThrottle;
            brake = currentBrake;
            break;
    }
    
    // Now determine steering based on the action group
    switch(actionGroup) {
        case 0: // NEUTRAL
            steering = 0.0f;
            break;
        case 1: // LEFT
            steering = 0.3f * car.config.maxSteeringAngle;
            break;
        case 2: // RIGHT
            steering = -0.3f * car.config.maxSteeringAngle;
            break;
    }
    
    // For NO_CHANGE action, also keep the current steering
    if (throttleBrakeAction == 8) {
        steering = currentSteering;
    }
    
    return std::make_tuple(throttle, brake, steering);
}

void DQNEnvironment::updateLaneInformation() {
    // Update lateral deviation (assuming lane center is at x=0)
    lateralDeviation = car.position.x;
    
    // Calculate lane heading error (assuming lane direction is along z-axis)
    float laneDirection = 0.0f; // Assuming lane points along z-axis (0 radians)
    laneHeadingError = car.rotation - laneDirection;
    
    // Normalize heading error to range [-PI, PI]
    while (laneHeadingError > PI) laneHeadingError -= 2 * PI;
    while (laneHeadingError < -PI) laneHeadingError += 2 * PI;
}

std::vector<float> DQNEnvironment::getState() {
    std::vector<float> state(9);
    
    // Original state variables
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
    
    // New lane following state variables
    // Lateral deviation from lane center (normalized)
    state[6] = lateralDeviation / (config.laneWidth / 2.0f);  // Normalized to [-1, 1] for lane width
    
    // Heading error (normalized)
    state[7] = laneHeadingError / PI;  // Normalized to [-1, 1]
    
    // Steering angle (normalized)
    state[8] = car.steeringAngle / car.config.maxSteeringAngle;  // Normalized to [-1, 1]
    
    return state;
}

float DQNEnvironment::calculateReward(const std::vector<float>& state, Action action) {
    float reward = 0.0f;
    
    // Speed control reward (from original implementation)
    float speedDiff = std::abs(car.speed - config.targetSpeed);
    
    if (speedDiff < config.speedRewardThreshold) {
        // Maximum reward when within threshold
        reward += 1.0f;
    } else {
        // Gradually decreasing reward as the difference increases
        reward += std::exp(-speedDiff * 0.5f);
    }
    
    // Lane following reward
    float lateralDeviationNormalized = std::abs(lateralDeviation) / (config.laneWidth / 2.0f);
    
    // Higher reward for staying close to center
    if (lateralDeviationNormalized < 0.2f) {
        reward += 1.0f; // Bonus for being very close to center
    } else {
        // Penalty increases exponentially as the car moves away from the center
        reward -= config.lateralDeviationPenalty * std::pow(lateralDeviationNormalized, 2);
    }
    
    // Heading alignment reward
    float headingErrorNormalized = std::abs(laneHeadingError) / PI;
    reward -= headingErrorNormalized * 0.5f; // Penalty for misalignment with lane
    
    // Penalize large changes in controls
    if (lastAction != action && 
        action != NO_CHANGE_NEUTRAL && action != NO_CHANGE_LEFT && action != NO_CHANGE_RIGHT && 
        lastAction != NO_CHANGE_NEUTRAL && lastAction != NO_CHANGE_LEFT && lastAction != NO_CHANGE_RIGHT) {
        // Only penalize if action changed significantly
        int lastActionGroup = lastAction / 9;
        int currentActionGroup = action / 9;
        int lastThrottleBrakeAction = lastAction % 9;
        int currentThrottleBrakeAction = action % 9;
        
        // Penalize large throttle/brake changes
        if (std::abs(currentThrottleBrakeAction - lastThrottleBrakeAction) > 2) {
            reward -= 0.2f;
        }
        
        // Penalize steering direction changes
        if (lastActionGroup != currentActionGroup) {
            reward -= 0.1f;
        }
    }
    
    // Penalize extreme throttle when too fast
    if (car.throttle > 0.8f && car.speed > config.targetSpeed * 1.1f) {
        reward -= 0.3f;
    }
    
    // Penalize unnecessary braking when too slow
    if (car.brake > 0.0f && car.speed < config.targetSpeed * 0.9f) {
        reward -= 0.3f;
    }
    
    return reward;
}

} // namespace CarGame