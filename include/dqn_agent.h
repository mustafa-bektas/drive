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
        int hiddenSize3;            // Third hidden layer size (optional, for larger networks)

        // Constructor with default values
        Config()
            : stateSize(9),         // Expanded state size for lane following
              actionSize(27),       // Expanded action space for steering
              hiddenSize1(128),     // Larger hidden layer for more complex behavior
              hiddenSize2(64),      // Larger second hidden layer
              hiddenSize3(0)        // Optional third layer (0 means not used)
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