#include "dqn_agent.h"
#include <algorithm>
#include <numeric>

namespace CarGame {

DQNAgent::DQNAgent(Config config)
    : config(config),
      epsilon(config.epsilonStart),
      lossSum(0.0f),
      updateCount(0),
      rng(std::random_device{}()) {
    
    // Create Q-network
    std::vector<int> layerSizes = {
        config.stateSize,
        config.hiddenSize1,
        config.hiddenSize2,
        config.actionSize
    };
    
    qNetwork = std::make_unique<NeuralNetwork>(layerSizes);
    targetNetwork = std::make_unique<NeuralNetwork>(layerSizes);
    
    // Copy initial weights to target network
    targetNetwork->copyWeightsFrom(*qNetwork);
    
    // Create optimizer
    optimizer = std::make_unique<DQNOptimizer>(*qNetwork, config.learningRate);
}

DQNEnvironment::Action DQNAgent::selectAction(const std::vector<float>& state, bool explore) {
    // Exploration: select random action with probability epsilon
    if (explore) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng) < epsilon) {
            std::uniform_int_distribution<int> actionDist(0, config.actionSize - 1);
            return static_cast<DQNEnvironment::Action>(actionDist(rng));
        }
    }
    
    // Exploitation: select best action based on Q-values
    return getBestAction(state);
}

void DQNAgent::addExperience(const std::vector<float>& state,
                             DQNEnvironment::Action action,
                             float reward,
                             const std::vector<float>& nextState,
                             bool done) {
    experiences.push_back({state, action, reward, nextState, done});
    
    // Limit buffer size
    if (experiences.size() > config.replayBufferSize) {
        experiences.pop_front();
    }
}

void DQNAgent::trainNetwork() {
    if (experiences.size() < config.batchSize) {
        return;  // Not enough samples for training
    }
    
    // Sample random batch from replay buffer
    std::vector<std::vector<float>> states;
    std::vector<DQNEnvironment::Action> actions;
    std::vector<float> rewards;
    std::vector<std::vector<float>> nextStates;
    std::vector<bool> dones;
    
    std::vector<int> indices(experiences.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    for (int i = 0; i < config.batchSize; ++i) {
        const auto& exp = experiences[indices[i]];
        states.push_back(exp.state);
        actions.push_back(exp.action);
        rewards.push_back(exp.reward);
        nextStates.push_back(exp.nextState);
        dones.push_back(exp.done);
    }
    
    // Calculate loss and gradients
    float loss = calculateLoss(states, actions, rewards, nextStates, dones);
    
    // Update loss history
    lossHistory.push_back(loss);
    lossSum += loss;
    if (lossHistory.size() > 100) {
        lossSum -= lossHistory.front();
        lossHistory.pop_front();
    }
    
    // Decay exploration rate
    epsilon = std::max(config.epsilonMin, epsilon * config.epsilonDecay);
    
    // Update target network if needed
    updateCount++;
    if (updateCount % config.targetUpdateFreq == 0) {
        updateTargetNetwork();
    }
}

void DQNAgent::updateTargetNetwork() {
    targetNetwork->softUpdateFrom(*qNetwork, config.targetUpdateRate);
}

std::vector<float> DQNAgent::getQValues(const std::vector<float>& state, bool useTarget) {
    if (useTarget) {
        return targetNetwork->forward(state);
    } else {
        return qNetwork->forward(state);
    }
}

DQNEnvironment::Action DQNAgent::getBestAction(const std::vector<float>& state) {
    auto qValues = getQValues(state);
    
    // Find action with highest Q-value
    auto maxIt = std::max_element(qValues.begin(), qValues.end());
    int bestActionIndex = std::distance(qValues.begin(), maxIt);
    
    return static_cast<DQNEnvironment::Action>(bestActionIndex);
}

float DQNAgent::calculateLoss(const std::vector<std::vector<float>>& states,
                             const std::vector<DQNEnvironment::Action>& actions,
                             const std::vector<float>& rewards,
                             const std::vector<std::vector<float>>& nextStates,
                             const std::vector<bool>& dones) {
    // Calculate target Q-values (y_j)
    std::vector<std::vector<float>> targets;
    
    for (size_t i = 0; i < states.size(); ++i) {
        // Get current Q-values for the state
        std::vector<float> currentQValues = getQValues(states[i]);
        std::vector<float> targetQValues = currentQValues;
        
        // Calculate target for the action taken
        if (dones[i]) {
            targetQValues[static_cast<int>(actions[i])] = rewards[i];
        } else {
            // Get max Q-value for next state from target network
            std::vector<float> nextQValues = getQValues(nextStates[i], true);
            float maxNextQ = *std::max_element(nextQValues.begin(), nextQValues.end());
            
            // Q-learning update: Q(s,a) = r + gamma * max_a' Q(s',a')
            targetQValues[static_cast<int>(actions[i])] = rewards[i] + config.gamma * maxNextQ;
        }
        
        targets.push_back(targetQValues);
    }
    
    // Calculate MSE loss
    float totalLoss = 0.0f;
    for (size_t i = 0; i < states.size(); ++i) {
        auto predictedQ = getQValues(states[i]);
        auto targetQ = targets[i];
        
        for (size_t j = 0; j < predictedQ.size(); ++j) {
            totalLoss += 0.5f * std::pow(predictedQ[j] - targetQ[j], 2);
        }
    }
    totalLoss /= states.size();
    
    // Update network weights
    optimizer->updateWeights(states, targets);
    
    return totalLoss;
}

} // namespace CarGame