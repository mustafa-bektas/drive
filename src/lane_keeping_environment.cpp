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
    // don't create new car here to avoid overriding
    // dqn car - just reset internal state
    
    // reset internal state
    currentStep = 0;
    lastAction = MAINTAIN_STEERING;
    
    // reset lane metrics
    lateralPosition = getLateralPosition(carRef);
    headingError = getHeadingError(carRef);
    lateralVelocity = 0.0f;
    
    return getState(carRef);
}

std::tuple<std::vector<float>, float, bool> LaneKeepingEnvironment::step(Action action, Car& car) {
    currentStep++;
    
    // apply steering adjustment
    float steeringAdjustment = actionToSteeringAdjustment(action);
    
    // return-to-center effect
    if (action == MAINTAIN_STEERING) {
        // 30% return rate per step
        car.steeringAngle *= 0.7f;
    } else {
        car.steeringAngle += steeringAdjustment;
    }
    
    // limit steering angle
    if (car.steeringAngle > car.config.maxSteeringAngle) {
        car.steeringAngle = car.config.maxSteeringAngle;
    } else if (car.steeringAngle < -car.config.maxSteeringAngle) {
        car.steeringAngle = -car.config.maxSteeringAngle;
    }
    
    // update lane metrics
    lateralPosition = getLateralPosition(car);
    headingError = getHeadingError(car);
    
    // get lateral velocity
    lateralVelocity = car.lateralVelocity;
    
    // get new state
    std::vector<float> state = getState(car);
    
    // calc reward
    float reward = calculateReward(state, action, car);
    
    // check termination
    bool done = false;
    
    // terminate if out of lane
    if (std::abs(lateralPosition) > config.maxLateralDeviation) {
        done = true;
        reward -= 10.0f; // penalty for leaving lane
    }
    
    // terminate after max steps
    if (currentStep >= config.maxEpisodeSteps) {
        done = true;
    }
    
    // save last action for reward
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
    
    // lat pos from center (normalized by lane width)
    state[0] = lateralPosition / (config.laneWidth / 2.0f);
    
    // heading err (normalized)
    state[1] = headingError / 1.0f;  // normalized by +-1 rad range
    
    // lat velocity (normalized)
    state[2] = lateralVelocity / 5.0f;  // typical max lat vel
    
    // steering angle (normalized by max)
    state[3] = car.steeringAngle / car.config.maxSteeringAngle;
    
    // distance to lane edge (normalized)
    float distanceToBoundary = (config.laneWidth / 2.0f) - std::abs(lateralPosition);
    state[4] = distanceToBoundary / (config.laneWidth / 2.0f);
    
    return state;
}

float LaneKeepingEnvironment::calculateReward(const std::vector<float>& state, Action action, const Car& car) {
    float reward = 0.0f;
    
    // reward for center
    float centeringReward = std::exp(-5.0f * std::abs(lateralPosition));
    reward += centeringReward * 2.0f;
    
    // reward for alignment
    float alignmentReward = 1.0f - std::abs(state[1]);  // 1.0 when aligned
    reward += alignmentReward;
    
    // penalize abrupt steering
    if (lastAction != action && 
        action != MAINTAIN_STEERING && 
        lastAction != MAINTAIN_STEERING) {
        int actionDiff = std::abs(static_cast<int>(action) - static_cast<int>(lastAction));
        if (actionDiff > 2) {
            reward -= 0.5f * static_cast<float>(actionDiff - 2);
        }
    }
    
    // penalize extreme steering
    float steeringRatio = std::abs(car.steeringAngle / car.config.maxSteeringAngle);
    if (steeringRatio > 0.8f) {
        reward -= 0.5f * (steeringRatio - 0.8f) / 0.2f;
    }
    
    return reward;
}

float LaneKeepingEnvironment::getLateralPosition(const Car& car) const {
    // lane centered at x=0 along z-axis
    return car.position.x;
}

float LaneKeepingEnvironment::getHeadingError(const Car& car) const {
    // lane direction along z-axis (0 rad)
    float laneDirection = 0.0f;
    
    // heading error normalized to [-PI, PI]
    float error = car.rotation - laneDirection;
    while (error > PI) error -= 2.0f * PI;
    while (error < -PI) error += 2.0f * PI;
    
    return error;
}

float LaneKeepingEnvironment::getLateralVelocity(const Car& car) const {
    // x velocity = lateral movement
    return car.velocity.x;
}

} // namespace CarGame