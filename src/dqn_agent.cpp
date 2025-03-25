#include "dqn_agent.h"
#include "model_loader.h"
#include <algorithm>
#include <iostream>

namespace CarGame {

DQNAgent::DQNAgent(Config config)
    : config(config),
      rng(std::random_device{}()) {
    
    // Create Q-network for inference
    std::vector<int> layerSizes = {
        config.stateSize,
        config.hiddenSize1,
        config.hiddenSize2,
        config.actionSize
    };
    
    network = std::make_unique<NeuralNetwork>(layerSizes);
}

DQNEnvironment::Action DQNAgent::selectAction(const std::vector<float>& state, bool explore) {
    // Exploration is disabled in inference-only mode, but kept as parameter for compatibility
    if (explore) {
        // For demo purposes, you might want to occasionally explore
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng) < 0.05f) { // 5% exploration rate
            std::uniform_int_distribution<int> actionDist(0, config.actionSize - 1);
            return static_cast<DQNEnvironment::Action>(actionDist(rng));
        }
    }
    
    // Exploitation: select best action based on Q-values
    return getBestAction(state);
}

DQNEnvironment::Action DQNAgent::getBestAction(const std::vector<float>& state) {
    // Forward pass through the network to get Q-values
    std::vector<float> qValues = network->forward(state);
    
    // Find action with highest Q-value
    auto maxIt = std::max_element(qValues.begin(), qValues.end());
    int bestActionIndex = std::distance(qValues.begin(), maxIt);
    
    return static_cast<DQNEnvironment::Action>(bestActionIndex);
}

bool DQNAgent::loadModel(const std::string& filename) {
    return ModelLoader::loadModelFromPython(filename, *network);
}

} // namespace CarGame