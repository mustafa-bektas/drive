#pragma once

#include "lane_keeping_environment.h"
#include "neural_network.h"
#include <vector>
#include <memory>
#include <random>

namespace CarGame {

class LaneKeepingAgent {
public:
    struct Config {
        int stateSize;              // state vec size
        int actionSize;             // num actions
        int hiddenSize1;            // first hidden size
        int hiddenSize2;            // second hidden size

        // defaults
        Config()
            : stateSize(5),
              actionSize(7),
              hiddenSize1(128),
              hiddenSize2(64)
        {}
    };
    
    LaneKeepingAgent(Config config = Config());
    
    // pick action for state
    LaneKeepingEnvironment::Action selectAction(const std::vector<float>& state);
    
    // load model
    bool loadModel(const std::string& filename);
    
    // direct access
    NeuralNetwork* getNetwork() const { return network.get(); }
    
private:
    Config config;
    std::unique_ptr<NeuralNetwork> network;
    std::mt19937 rng;
    
    // get best action for inference
    LaneKeepingEnvironment::Action getBestAction(const std::vector<float>& state);
};

} // namespace CarGame