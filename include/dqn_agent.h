#pragma once

#include "dqn_environment.h"
#include "neural_network.h"
#include <vector>
#include <deque>
#include <random>
#include <memory>

namespace CarGame {

class DQNAgent {
public:
    struct Experience {
        std::vector<float> state;
        DQNEnvironment::Action action;
        float reward;
        std::vector<float> nextState;
        bool done;
    };
    
    struct Config {
        int stateSize;              // Size of state vector
        int actionSize;             // Number of possible actions
        int hiddenSize1;            // First hidden layer size
        int hiddenSize2;            // Second hidden layer size
        int batchSize;              // Training batch size
        int replayBufferSize;       // Experience replay buffer capacity
        float gamma;                // Discount factor
        float epsilonStart;         // Starting exploration rate
        float epsilonMin;           // Minimum exploration rate
        float epsilonDecay;         // Exploration decay rate
        float learningRate;         // Learning rate for optimizer
        float targetUpdateRate;     // Soft update rate for target network
        int targetUpdateFreq;       // Steps between target network updates

        // Constructor with default values
        Config()
            : stateSize(6),
              actionSize(9),
              hiddenSize1(64),
              hiddenSize2(32),
              batchSize(32),
              replayBufferSize(10000),
              gamma(0.99f),
              epsilonStart(1.0f),
              epsilonMin(0.1f),
              epsilonDecay(0.995f),
              learningRate(0.001f),
              targetUpdateRate(0.01f),
              targetUpdateFreq(10)
        {}
    };
    
    DQNAgent(Config config = Config());
    
    // Select an action based on current state
    DQNEnvironment::Action selectAction(const std::vector<float>& state, bool explore = true);
    
    // Add experience to replay buffer
    void addExperience(const std::vector<float>& state, 
                       DQNEnvironment::Action action,
                       float reward,
                       const std::vector<float>& nextState,
                       bool done);
    
    // Train the network on a batch from the replay buffer
    void trainNetwork();
    
    // Update target network
    void updateTargetNetwork();
    
    // Getters for monitoring
    float getExplorationRate() const { return epsilon; }
    float getAverageLoss() const { return lossHistory.empty() ? 0.0f : lossSum / lossHistory.size(); }
    int getUpdateCount() const { return updateCount; }
    const std::deque<float>& getLossHistory() const { return lossHistory; }
    bool canTrain() const { return experiences.size() >= config.batchSize; }
    
    // Get network for saving/loading (needed for headless training)
    NeuralNetwork* getQNetwork() const { return qNetwork.get(); }
    
private:
    Config config;
    std::unique_ptr<NeuralNetwork> qNetwork;
    std::unique_ptr<NeuralNetwork> targetNetwork;
    std::unique_ptr<DQNOptimizer> optimizer;
    
    std::deque<Experience> experiences;
    std::deque<float> lossHistory;
    
    float epsilon;
    float lossSum;
    int updateCount;
    
    std::mt19937 rng;
    
    // Get Q values for a state
    std::vector<float> getQValues(const std::vector<float>& state, bool useTarget = false);
    
    // Get best action for a state
    DQNEnvironment::Action getBestAction(const std::vector<float>& state);
    
    // Calculate loss for one training step
    float calculateLoss(const std::vector<std::vector<float>>& states,
                       const std::vector<DQNEnvironment::Action>& actions,
                       const std::vector<float>& rewards,
                       const std::vector<std::vector<float>>& nextStates,
                       const std::vector<bool>& dones);
};

} // namespace CarGame