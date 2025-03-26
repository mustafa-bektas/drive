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
    // random pos within lane
    float laneWidth = 10.0f; // match lane width in rendering.cpp
    float randomLateralPosition = ((float)rand() / RAND_MAX - 0.5f) * laneWidth * 1.02f;
    
    // slight heading variation
    float randomRotation = ((float)rand() / RAND_MAX - 0.5f) * 0.2f; // ±0.1 radians
    
    car = Car(Vector3{randomLateralPosition, 0.5f, 0.0f});
    car.rotation = randomRotation;
    currentStep = 0;
    
    // reset controls for NO_CHANGE action
    currentThrottle = 0.0f;
    currentBrake = 0.0f;
    lastAction = COAST;
    
    return getState();
}

std::tuple<std::vector<float>, float, bool> DQNEnvironment::step(Action action) {
    // apply action to car
    std::pair<float, float> controls = actionToControls(action);
    car.throttle = controls.first;
    car.brake = controls.second;
    
    // save controls for NO_CHANGE action
    currentThrottle = controls.first;
    currentBrake = controls.second;
    
    // update physics
    car.update(config.timeStep);
    currentStep++;
    
    // get new state
    std::vector<float> newState = getState();
    
    // calc reward
    float reward = calculateReward(newState, action);
    
    // check if done
    bool done = currentStep >= config.maxEpisodeSteps;
    
    // remember last action for reward
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
            // both zero
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
    
    // speed (normalized)
    state[0] = car.speed / 40.0f;  // max 40 m/s
    
    // speed diff from target (normalized)
    state[1] = (car.speed - config.targetSpeed) / 40.0f;
    
    // accel (normalized)
    state[2] = car.acceleration.x / 10.0f;  // max 10 m/s2
    
    // throttle
    state[3] = car.throttle;
    
    // brake
    state[4] = car.brake;
    
    // rpm (normalized)
    state[5] = car.engineSpeed / 8000.0f;  // based on max in car.cpp
    
    return state;
}

float DQNEnvironment::calculateReward(const std::vector<float>& state, Action action) {
    float reward = 0.0f;
    
    // main reward: closeness to target speed
    float speedDiff = std::abs(car.speed - config.targetSpeed);
    
    if (speedDiff < config.speedRewardThreshold) {
        // max reward within threshold
        reward += 1.0f;
    } else {
        // decreasing reward as diff increases
        reward += std::exp(-speedDiff * 0.5f);
    }
    
    // penalize big control changes
    if (lastAction != action && 
        action != NO_CHANGE && 
        lastAction != NO_CHANGE) {
        // only penalize significant changes
        if (std::abs(static_cast<int>(action) - static_cast<int>(lastAction)) > 2) {
            reward -= 0.2f;
        }
    }
    
    // penalize excessive throttle
    if (car.throttle > 0.8f && car.speed > config.targetSpeed * 1.1f) {
        reward -= 0.3f;
    }
    
    // penalize unneeded braking
    if (car.brake > 0.0f && car.speed < config.targetSpeed * 0.9f) {
        reward -= 0.3f;
    }
    
    return reward;
}

} // namespace CarGame