#pragma once

#include "dqn_environment.h"
#include "neural_network.h"
#include <vector>
#include <memory>
#include <random>

namespace CarGame {

// dqn agent - inference only
class DQNAgent {
public:
    struct Config {
        int stateSize;              // state vec size
        int actionSize;             // num of actions
        int hiddenSize1;            // first hidden layer
        int hiddenSize2;            // second hidden layer

        // defaults
        Config()
            : stateSize(6),
              actionSize(9),
              hiddenSize1(128),
              hiddenSize2(64)
        {}
    };
    
    DQNAgent(Config config = Config());
    
    // choose action based on state
    DQNEnvironment::Action selectAction(const std::vector<float>& state);
    
    // load from file
    bool loadModel(const std::string& filename);
    
    // direct access if needed
    NeuralNetwork* getNetwork() const { return network.get(); }
    
private:
    Config config;
    std::unique_ptr<NeuralNetwork> network;
    std::mt19937 rng;
    
    // get best action for inference
    DQNEnvironment::Action getBestAction(const std::vector<float>& state);
};

} // namespace CarGame