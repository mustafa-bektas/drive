#include "rl_agent.h"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <fstream>

namespace CarGame {

RLAgent::RLAgent(float learningRate, float discountFactor, float explorationRate)
    : alpha(learningRate), gamma(discountFactor), epsilon(explorationRate), gen(rd()) {
}

std::vector<float> RLAgent::getAction(const std::vector<float>& state) {
    // Discretize the state for table lookup
    std::vector<float> discreteState = discretizeState(state);
    std::string stateKey = stateToString(discreteState);
    
    // Exploration: random action with probability epsilon
    std::uniform_real_distribution<> distr(0.0, 1.0);
    if (distr(gen) < epsilon) {
        // Return random action [throttle, brake]
        return {static_cast<float>(distr(gen)), static_cast<float>(distr(gen) * 0.5f)}; // Less random brake to avoid jerky behavior
    }
    
    // Exploitation: best known action
    if (qTable.find(stateKey) == qTable.end()) {
        // State not seen before, initialize with zero values
        qTable[stateKey] = std::map<std::string, float>();
    }
    
    // If no actions exist yet, use a default action
    if (qTable[stateKey].empty()) {
        return {0.5f, 0.0f}; // Default: moderate throttle, no brake
    }
    
    // Find action with highest Q-value
    std::string bestActionKey;
    float bestValue = -std::numeric_limits<float>::max();
    
    for (const auto& actionPair : qTable[stateKey]) {
        if (actionPair.second > bestValue) {
            bestValue = actionPair.second;
            bestActionKey = actionPair.first;
        }
    }
    
    // Parse best action string back to vector
    std::vector<float> bestAction;
    std::istringstream iss(bestActionKey);
    std::string token;
    while (std::getline(iss, token, ',')) {
        bestAction.push_back(std::stof(token));
    }
    
    return bestAction;
}

void RLAgent::updateQValues(const std::vector<float>& state, const std::vector<float>& action, 
                           float reward, const std::vector<float>& nextState) {
    // Discretize states
    std::vector<float> discreteState = discretizeState(state);
    std::vector<float> discreteNextState = discretizeState(nextState);
    
    std::string stateKey = stateToString(discreteState);
    std::string actionKey = actionToString(action);
    std::string nextStateKey = stateToString(discreteNextState);
    
    // Initialize Q-value if not exists
    if (qTable.find(stateKey) == qTable.end()) {
        qTable[stateKey] = std::map<std::string, float>();
    }
    if (qTable[stateKey].find(actionKey) == qTable[stateKey].end()) {
        qTable[stateKey][actionKey] = 0.0f;
    }
    
    // Get current Q value
    float currentQ = qTable[stateKey][actionKey];
    
    // Find max Q value for next state
    float maxNextQ = 0.0f;
    if (qTable.find(nextStateKey) != qTable.end() && !qTable[nextStateKey].empty()) {
        maxNextQ = -std::numeric_limits<float>::max();
        for (const auto& actionPair : qTable[nextStateKey]) {
            maxNextQ = std::max(maxNextQ, actionPair.second);
        }
    }
    
    // Q-learning update
    float newQ = currentQ + alpha * (reward + gamma * maxNextQ - currentQ);
    qTable[stateKey][actionKey] = newQ;
}

std::string RLAgent::stateToString(const std::vector<float>& state) {
    std::ostringstream oss;
    for (size_t i = 0; i < state.size(); ++i) {
        if (i > 0) oss << ";";
        oss << state[i];
    }
    return oss.str();
}

std::string RLAgent::actionToString(const std::vector<float>& action) {
    std::ostringstream oss;
    for (size_t i = 0; i < action.size(); ++i) {
        if (i > 0) oss << ",";
        oss << action[i];
    }
    return oss.str();
}

std::vector<float> RLAgent::discretizeState(const std::vector<float>& state) {
    // Discretize continuous state space for Q-table lookup
    std::vector<float> discretized;
    for (float value : state) {
        // Round to nearest 0.5 to reduce state space
        discretized.push_back(std::round(value * 2.0f) / 2.0f);
    }
    return discretized;
}

void RLAgent::saveModel(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        printf("Error: Unable to open file for saving model: %s\n", filename.c_str());
        return;
    }
    
    // Save parameters
    file.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
    file.write(reinterpret_cast<const char*>(&gamma), sizeof(float));
    file.write(reinterpret_cast<const char*>(&epsilon), sizeof(float));
    
    // Save Q-table size
    size_t stateCount = qTable.size();
    file.write(reinterpret_cast<const char*>(&stateCount), sizeof(size_t));
    
    // Save each state and its actions
    for (const auto& statePair : qTable) {
        // Write state key
        size_t stateKeyLength = statePair.first.length();
        file.write(reinterpret_cast<const char*>(&stateKeyLength), sizeof(size_t));
        file.write(statePair.first.c_str(), stateKeyLength);
        
        // Write number of actions for this state
        size_t actionCount = statePair.second.size();
        file.write(reinterpret_cast<const char*>(&actionCount), sizeof(size_t));
        
        // Write each action and its Q-value
        for (const auto& actionPair : statePair.second) {
            // Write action key
            size_t actionKeyLength = actionPair.first.length();
            file.write(reinterpret_cast<const char*>(&actionKeyLength), sizeof(size_t));
            file.write(actionPair.first.c_str(), actionKeyLength);
            
            // Write Q-value
            file.write(reinterpret_cast<const char*>(&actionPair.second), sizeof(float));
        }
    }
    
    file.close();
    printf("Model saved to: %s\n", filename.c_str());
}

bool RLAgent::loadModel(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        printf("Error: Unable to open file for loading model: %s\n", filename.c_str());
        return false;
    }
    
    // Clear existing Q-table
    qTable.clear();
    
    // Load parameters
    file.read(reinterpret_cast<char*>(&alpha), sizeof(float));
    file.read(reinterpret_cast<char*>(&gamma), sizeof(float));
    file.read(reinterpret_cast<char*>(&epsilon), sizeof(float));
    
    // Load Q-table size
    size_t stateCount;
    file.read(reinterpret_cast<char*>(&stateCount), sizeof(size_t));
    
    // Load each state and its actions
    for (size_t i = 0; i < stateCount; i++) {
        // Read state key
        size_t stateKeyLength;
        file.read(reinterpret_cast<char*>(&stateKeyLength), sizeof(size_t));
        
        std::string stateKey(stateKeyLength, ' ');
        file.read(&stateKey[0], stateKeyLength);
        
        // Create new map for this state
        qTable[stateKey] = std::map<std::string, float>();
        
        // Read number of actions for this state
        size_t actionCount;
        file.read(reinterpret_cast<char*>(&actionCount), sizeof(size_t));
        
        // Read each action and its Q-value
        for (size_t j = 0; j < actionCount; j++) {
            // Read action key
            size_t actionKeyLength;
            file.read(reinterpret_cast<char*>(&actionKeyLength), sizeof(size_t));
            
            std::string actionKey(actionKeyLength, ' ');
            file.read(&actionKey[0], actionKeyLength);
            
            // Read Q-value
            float qValue;
            file.read(reinterpret_cast<char*>(&qValue), sizeof(float));
            
            // Store in Q-table
            qTable[stateKey][actionKey] = qValue;
        }
    }
    
    file.close();
    printf("Model loaded from: %s\n", filename.c_str());
    return true;
}

float RLAgent::getAverageQValue() const {
    float sum = 0.0f;
    int count = 0;
    
    for (const auto& statePair : qTable) {
        for (const auto& actionPair : statePair.second) {
            sum += actionPair.second;
            count++;
        }
    }
    
    return count > 0 ? sum / count : 0.0f;
}


} // namespace CarGame