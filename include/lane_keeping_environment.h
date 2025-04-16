#pragma once

#include "car.h"
#include <vector>
#include <random>
#include <tuple>
#include <memory>

namespace CarGame {

class LaneKeepingEnvironment {
public:
    // lane keeping config
    struct Config {
        float laneWidth;               // lane width in meters
        float maxLateralDeviation;     // max allowed deviation
        float maxEpisodeSteps;         // max steps per episode
        float timeStep;                // sim time step
        int actionSpace;               // num discrete actions
        
        // defaults
        Config() 
            : laneWidth(7.5f), // Reduced from 10.0f
              maxLateralDeviation(3.75f), // Also reduce max deviation proportionally (0.75 * 5.0f)
              maxEpisodeSteps(1000),
              timeStep(1.0f / 60.0f),
              actionSpace(7)
        {}
    };

    // discrete actions for dqn
    enum Action {
        TURN_HARD_LEFT = 0,     // big left turn
        TURN_MEDIUM_LEFT = 1,   // medium left
        TURN_GENTLE_LEFT = 2,   // small left
        MAINTAIN_STEERING = 3,  // no change
        TURN_GENTLE_RIGHT = 4,  // small right
        TURN_MEDIUM_RIGHT = 5,  // medium right
        TURN_HARD_RIGHT = 6     // big right
    };

    LaneKeepingEnvironment(Config config = Config());
    
    // reset env
    std::vector<float> reset(const Car& carRef);
    
    // step based on action
    std::tuple<std::vector<float>, float, bool> step(Action action, Car& car);
    
    // convert action to steering change
    float actionToSteeringAdjustment(Action action);
    
    // helpers
    int getStateSize() const { return 5; } // state size
    int getActionSize() const { return config.actionSpace; }
    
    // lane position info
    float getLateralPosition(const Car& car) const;
    float getHeadingError(const Car& car) const;
    float getLateralVelocity(const Car& car) const;
    
private:
    Config config;
    int currentStep;
    std::mt19937 rng;
    
    // last action for smoothness
    Action lastAction;
    
    // current lateral state
    float lateralPosition; // dist from center
    float headingError;    // heading vs lane dir
    float lateralVelocity; // lateral position change rate
    
    std::vector<float> getState(const Car& car);
    float calculateReward(const std::vector<float>& state, Action action, const Car& car);
};

} // namespace CarGame
