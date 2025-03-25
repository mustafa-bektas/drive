#include "lane_keeping_agent.h"
#include "model_loader.h"
#include <algorithm>
#include <iostream>

namespace CarGame {

LaneKeepingAgent::LaneKeepingAgent(Config config)
    : config(config),
      rng(std::random_device{}()) {
    
    // Create neural network for inference
    std::vector<int> layerSizes = {
        config.stateSize,
        config.hiddenSize1,
        config.hiddenSize2,
        config.actionSize
    };
    
    network = std::make_unique<NeuralNetwork>(layerSizes);
}

LaneKeepingEnvironment::Action LaneKeepingAgent::selectAction(const std::vector<float>& state) {
    // Inference only - select best action based on Q-values
    return getBestAction(state);
}

LaneKeepingEnvironment::Action LaneKeepingAgent::getBestAction(const std::vector<float>& state) {
    // Forward pass through the network to get Q-values
    std::vector<float> qValues = network->forward(state);
    
    // Find action with highest Q-value
    auto maxIt = std::max_element(qValues.begin(), qValues.end());
    int bestActionIndex = std::distance(qValues.begin(), maxIt);
    
    return static_cast<LaneKeepingEnvironment::Action>(bestActionIndex);
}

bool LaneKeepingAgent::loadModel(const std::string& filename) {
    return ModelLoader::loadModelFromPython(filename, *network);
}

} // namespace CarGame