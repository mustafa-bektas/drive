#pragma once

#include "lane_keeping_environment.h"
#include "neural_network.h"
#include <vector>
#include <memory>
#include <random>

namespace CarGame {

/**
 * Lane Keeping Agent for inference of trained models
 * Simplified to only include functionality needed for model inference
 */
class LaneKeepingAgent {
public:
    struct Config {
        int stateSize;              // Size of state vector
        int actionSize;             // Number of possible actions
        int hiddenSize1;            // First hidden layer size
        int hiddenSize2;            // Second hidden layer size

        // Constructor with default values
        Config()
            : stateSize(5),
              actionSize(7),
              hiddenSize1(64),
              hiddenSize2(32)
        {}
    };
    
    LaneKeepingAgent(Config config = Config());
    
    // Select an action based on current state
    LaneKeepingEnvironment::Action selectAction(const std::vector<float>& state);
    
    // Load model from file
    bool loadModel(const std::string& filename);
    
    // Get network for direct access (if needed)
    NeuralNetwork* getNetwork() const { return network.get(); }
    
private:
    Config config;
    std::unique_ptr<NeuralNetwork> network;
    std::mt19937 rng;
    
    // Get best action for a state (used in inference)
    LaneKeepingEnvironment::Action getBestAction(const std::vector<float>& state);
};

} // namespace CarGame