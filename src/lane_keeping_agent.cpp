#include "lane_keeping_agent.h"
#include "model_loader.h"
#include <algorithm>
#include <iostream>

namespace CarGame {

LaneKeepingAgent::LaneKeepingAgent(Config config)
    : config(config),
      rng(std::random_device{}()) {
    
    // create nn for inference
    std::vector<int> layerSizes = {
        config.stateSize,
        config.hiddenSize1,
        config.hiddenSize2,
        config.actionSize
    };
    
    network = std::make_unique<NeuralNetwork>(layerSizes);
}

LaneKeepingEnvironment::Action LaneKeepingAgent::selectAction(const std::vector<float>& state) {
    // pick best action
    return getBestAction(state);
}

LaneKeepingEnvironment::Action LaneKeepingAgent::getBestAction(const std::vector<float>& state) {
    // get q-vals
    std::vector<float> qValues = network->forward(state);
    
    // find highest q-val
    auto maxIt = std::max_element(qValues.begin(), qValues.end());
    int bestActionIndex = std::distance(qValues.begin(), maxIt);
    
    return static_cast<LaneKeepingEnvironment::Action>(bestActionIndex);
}

bool LaneKeepingAgent::loadModel(const std::string& filename) {
    return ModelLoader::loadModelFromPython(filename, *network);
}

} // namespace CarGame