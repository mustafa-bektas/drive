#include "dqn_agent.h"
#include "model_loader.h"
#include <algorithm>
#include <iostream>

namespace CarGame {

DQNAgent::DQNAgent(Config config)
    : config(config),
      rng(std::random_device{}()) {
    
    // setup q-network for inference
    std::vector<int> layerSizes = {
        config.stateSize,
        config.hiddenSize1,
        config.hiddenSize2,
        config.actionSize
    };
    
    network = std::make_unique<NeuralNetwork>(layerSizes);
}

DQNEnvironment::Action DQNAgent::selectAction(const std::vector<float>& state) {
    // just pick best action for inference
    return getBestAction(state);
}

DQNEnvironment::Action DQNAgent::getBestAction(const std::vector<float>& state) {
    // forward pass to get q-values
    std::vector<float> qValues = network->forward(state);
    
    // find action with highest q-value
    auto maxIt = std::max_element(qValues.begin(), qValues.end());
    int bestActionIndex = std::distance(qValues.begin(), maxIt);
    
    return static_cast<DQNEnvironment::Action>(bestActionIndex);
}

bool DQNAgent::loadModel(const std::string& filename) {
    return ModelLoader::loadModelFromPython(filename, *network);
}

} // namespace CarGame