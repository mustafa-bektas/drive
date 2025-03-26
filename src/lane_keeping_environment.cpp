#include "lane_keeping_environment.h"
#include <algorithm>
#include <cmath>

namespace CarGame {

LaneKeepingEnvironment::LaneKeepingEnvironment(Config config) 
    : config(config), 
      currentStep(0),
      rng(std::random_device{}()),
      lastAction(MAINTAIN_STEERING),
      lateralPosition(0.0f),
      headingError(0.0f),
      lateralVelocity(0.0f) {
}

std::vector<float> LaneKeepingEnvironment::reset(const Car& carRef) {
    // We don't create a new car in LaneKeepingEnvironment to avoid
    // overriding the car from DQNEnvironment, but we still need to
    // reset our internal state based on the car's new position
    
    // Reset internal state
    currentStep = 0;
    lastAction = MAINTAIN_STEERING;
    
    // Reset lane position metrics
    lateralPosition = getLateralPosition(carRef);
    headingError = getHeadingError(carRef);
    lateralVelocity = 0.0f;
    
    return getState(carRef);
}

std::tuple<std::vector<float>, float, bool> LaneKeepingEnvironment::step(Action action, Car& car) {
    currentStep++;
    
    // Apply steering adjustment based on action
    float steeringAdjustment = actionToSteeringAdjustment(action);
    
    // Add a steering return-to-center effect (natural steering behavior)
    // Only apply this when not actively steering hard
    if (action == MAINTAIN_STEERING) {
        // Return to center at 30% rate per step
        car.steeringAngle *= 0.7f;
    } else {
        car.steeringAngle += steeringAdjustment;
    }
    
    // Limit steering angle to the car's configuration limits
    if (car.steeringAngle > car.config.maxSteeringAngle) {
        car.steeringAngle = car.config.maxSteeringAngle;
    } else if (car.steeringAngle < -car.config.maxSteeringAngle) {
        car.steeringAngle = -car.config.maxSteeringAngle;
    }
    
    // Update lane position metrics
    lateralPosition = getLateralPosition(car);
    headingError = getHeadingError(car);
    
    // Calculate lateral velocity (change in lateral position)
    lateralVelocity = car.lateralVelocity;
    
    // Get new state
    std::vector<float> state = getState(car);
    
    // Calculate reward
    float reward = calculateReward(state, action, car);
    
    // Check for episode termination conditions
    bool done = false;
    
    // Terminate if car leaves the lane by too much
    if (std::abs(lateralPosition) > config.maxLateralDeviation) {
        done = true;
        reward -= 10.0f; // Extra penalty for leaving the lane
    }
    
    // Terminate after max steps
    if (currentStep >= config.maxEpisodeSteps) {
        done = true;
    }
    
    // Store last action for next reward calculation
    lastAction = action;
    
    return std::make_tuple(state, reward, done);
}
float LaneKeepingEnvironment::actionToSteeringAdjustment(Action action) {
    const float hardTurn = 0.05f;
    const float mediumTurn = 0.025f;
    const float gentleTurn = 0.01f;
    
    switch(action) {
        case TURN_HARD_LEFT:
            return hardTurn;
        case TURN_MEDIUM_LEFT:
            return mediumTurn;
        case TURN_GENTLE_LEFT:
            return gentleTurn;
        case MAINTAIN_STEERING:
            return 0.0f;
        case TURN_GENTLE_RIGHT:
            return -gentleTurn;
        case TURN_MEDIUM_RIGHT:
            return -mediumTurn;
        case TURN_HARD_RIGHT:
            return -hardTurn;
        default:
            return 0.0f;
    }
}

std::vector<float> LaneKeepingEnvironment::getState(const Car& car) {
    std::vector<float> state(5);
    
    // Lateral position from lane center (normalized by lane width)
    state[0] = lateralPosition / (config.laneWidth / 2.0f);
    
    // Heading error (normalized)
    state[1] = headingError / 1.0f;  // Normalized by a typical range of ±1 radian
    
    // Lateral velocity (normalized)
    state[2] = lateralVelocity / 5.0f;  // Normalized by typical max lateral velocity
    
    // Current steering angle (normalized by max steering angle)
    state[3] = car.steeringAngle / car.config.maxSteeringAngle;
    
    // Distance to nearest lane boundary (normalized by lane width)
    float distanceToBoundary = (config.laneWidth / 2.0f) - std::abs(lateralPosition);
    state[4] = distanceToBoundary / (config.laneWidth / 2.0f);
    
    return state;
}

float LaneKeepingEnvironment::calculateReward(const std::vector<float>& state, Action action, const Car& car) {
    float reward = 0.0f;
    
    // Reward for staying in the center of the lane
    float centeringReward = std::exp(-2.0f * std::abs(lateralPosition));
    reward += centeringReward * 2.0f;
    
    // Reward for aligning with the lane direction
    float alignmentReward = 1.0f - std::abs(state[1]);  // 1.0 when aligned, 0.0 at ±1 radian
    reward += alignmentReward;
    
    // Penalize abrupt steering changes
    if (lastAction != action && 
        action != MAINTAIN_STEERING && 
        lastAction != MAINTAIN_STEERING) {
        int actionDiff = std::abs(static_cast<int>(action) - static_cast<int>(lastAction));
        if (actionDiff > 2) {
            reward -= 0.5f * static_cast<float>(actionDiff - 2);
        }
    }
    
    // Penalize excessive steering angles
    float steeringRatio = std::abs(car.steeringAngle / car.config.maxSteeringAngle);
    if (steeringRatio > 0.8f) {
        reward -= 0.5f * (steeringRatio - 0.8f) / 0.2f;
    }
    
    return reward;
}

float LaneKeepingEnvironment::getLateralPosition(const Car& car) const {
    // In the current implementation, the lane is centered at x=0
    // and extends along the z-axis
    return car.position.x;
}

float LaneKeepingEnvironment::getHeadingError(const Car& car) const {
    // Lane direction is along the z-axis, so error is the difference
    // between car rotation and 0 (or PI, depending on direction)
    float laneDirection = 0.0f;  // Lane points along z-axis
    
    // Calculate heading error, normalizing to range [-PI, PI]
    float error = car.rotation - laneDirection;
    while (error > PI) error -= 2.0f * PI;
    while (error < -PI) error += 2.0f * PI;
    
    return error;
}

float LaneKeepingEnvironment::getLateralVelocity(const Car& car) const {
    // The x-component of velocity represents lateral movement
    return car.velocity.x;
}

} // namespace CarGame