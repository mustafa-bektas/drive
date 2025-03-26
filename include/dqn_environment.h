#pragma once

#include "car.h"
#include <vector>
#include <random>
#include <tuple>
#include <memory>

namespace CarGame {

class DQNEnvironment {
public:
    // rl env config
    struct Config {
        float targetSpeed;        // target in m/s
        float maxEpisodeSteps;    // max steps per episode
        float speedRewardThreshold; // threshold for max reward
        float timeStep;           // sim time step
        int actionSpace;          // num discrete actions
        
        // defaults
        Config() 
            : targetSpeed(50.0f / 3.6f),
              maxEpisodeSteps(1000),
              speedRewardThreshold(1.0f),
              timeStep(1.0f / 60.0f),
              actionSpace(9)
        {}
    };

    // discrete actions for dqn
    enum Action {
        STRONG_BRAKE = 0,    // 1.0
        MEDIUM_BRAKE = 1,    // 0.66
        LIGHT_BRAKE = 2,     // 0.33
        COAST = 3,           // nothing
        LIGHT_THROTTLE = 4,  // 0.25
        MEDIUM_THROTTLE = 5, // 0.5
        STRONG_THROTTLE = 6, // 0.75
        FULL_THROTTLE = 7,   // 1.0
        NO_CHANGE = 8        // keep current
    };

    DQNEnvironment(Config config = Config());
    
    // init/reset env
    std::vector<float> reset();
    
    // step sim based on action
    std::tuple<std::vector<float>, float, bool> step(Action action);
    
    // convert enum to actual controls
    std::pair<float, float> actionToControls(Action action);
    
    // get car for viz
    const Car& getCar() const { return car; }

    // non-const access
    Car& getCar() { return car; }
    
    // helpers for viz
    float getTargetSpeed() const { return config.targetSpeed; }
    int getStateSize() const { return 6; } // state vec size
    int getActionSize() const { return config.actionSpace; }
    
private:
    Car car;
    Config config;
    int currentStep;
    std::mt19937 rng;
    
    // remember last action
    Action lastAction;
    
    // current controls for NO_CHANGE action
    float currentThrottle;
    float currentBrake;
    
    std::vector<float> getState();
    float calculateReward(const std::vector<float>& state, Action action);
};

} // namespace CarGame