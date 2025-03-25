#pragma once

#include "dqn_environment.h"
#include "neural_network.h"
#include <vector>
#include <memory>
#include <random>

namespace CarGame {

/**
 * DQN Agent for inference of trained models
 * Simplified to only include functionality needed for model inference
 */
class DQNAgent {
public:
    struct Config {
        int stateSize;              // Size of state vector
        int actionSize;             // Number of possible actions
        int hiddenSize1;            // First hidden layer size
        int hiddenSize2;            // Second hidden layer size

        // Constructor with default values
        Config()
            : stateSize(6),
              actionSize(9),
              hiddenSize1(64),
              hiddenSize2(32)
        {}
    };
    
    DQNAgent(Config config = Config());
    
    // Select an action based on current state
    DQNEnvironment::Action selectAction(const std::vector<float>& state);
    
    // Load model from file
    bool loadModel(const std::string& filename);
    
    // Get network for direct access (if needed)
    NeuralNetwork* getNetwork() const { return network.get(); }
    
private:
    Config config;
    std::unique_ptr<NeuralNetwork> network;
    std::mt19937 rng;
    
    // Get best action for a state (used in inference)
    DQNEnvironment::Action getBestAction(const std::vector<float>& state);
};

} // namespace CarGame