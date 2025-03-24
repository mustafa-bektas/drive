#include "../include/rl_agent.h"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace CarGame {

    RLAgent::RLAgent(float learningRate, float discountFactor, float explorationRate)
    : alpha(learningRate), gamma(discountFactor), epsilon(explorationRate), 
      initialEpsilon(explorationRate), gen(rd()) {
    
    // ULTRASHORT ACTION SPACE - Just 3 critical actions
    actionSpace = {
        {0.0f, 0.0f},   // No throttle, no brake
        {0.4f, 0.0f},   // Moderate throttle
        {0.0f, 0.4f}    // Moderate brake
    };
    
    // Initialize Q-table with highly optimistic values
    for (int i = 0; i < 50; i += 2) { // For different speeds
        for (int j = -10; j <= 10; j += 2) { // For different errors
            std::string stateKey = std::to_string(i) + ";" + std::to_string(j);
            
            // Start with optimistic values for throttle when below target
            if (i < 8) { // Below target speed
                qTable[stateKey][actionToString({0.4f, 0.0f})] = 100.0f;  // Prefer throttle
                qTable[stateKey][actionToString({0.0f, 0.0f})] = 50.0f;   // Neutral
                qTable[stateKey][actionToString({0.0f, 0.4f})] = -50.0f;  // Avoid brake
            } 
            // Start with optimistic values for brake when above target
            else if (i > 10) { // Above target speed 
                qTable[stateKey][actionToString({0.0f, 0.4f})] = 100.0f;  // Prefer brake
                qTable[stateKey][actionToString({0.0f, 0.0f})] = 50.0f;   // Neutral
                qTable[stateKey][actionToString({0.4f, 0.0f})] = -50.0f;  // Avoid throttle
            }
            // Near target speed, prefer doing nothing
            else {
                qTable[stateKey][actionToString({0.0f, 0.0f})] = 100.0f;  // Prefer neutral
                qTable[stateKey][actionToString({0.4f, 0.0f})] = 0.0f;    // Throttle sometimes
                qTable[stateKey][actionToString({0.0f, 0.4f})] = 0.0f;    // Brake sometimes
            }
        }
    }
    
    printf("Q-table initialized with common-sense values\n");
}

std::vector<float> RLAgent::getAction(const std::vector<float>& state) {
    // Discretize the state for table lookup
    std::vector<float> discreteState = discretizeState(state);
    std::string stateKey = stateToString(discreteState);
    
    // Print current state for debugging
    if (state.size() >= 2) {
        //printf("State: Speed=%.1f, Error=%.1f | ", state[0], state[1]);
    }
    
    // Exploration: random action with probability epsilon
    std::uniform_real_distribution<> distr(0.0, 1.0);
    if (distr(gen) < epsilon) {
        // Use standard random selection
        int randomActionIdx = std::uniform_int_distribution<>(0, actionSpace.size() - 1)(gen);
        //printf("EXPLORE-RANDOM: %s\n", actionToString(actionSpace[randomActionIdx]).c_str());
        return actionSpace[randomActionIdx];
    }
    
    // Exploitation: best known action
    if (qTable.find(stateKey) == qTable.end()) {
        // State not seen before, initialize with optimistic values
        qTable[stateKey] = std::map<std::string, float>();
        if (state.size() >= 2) {
            float error = state[1];
            if (error < -1.0f) { // Too slow
                qTable[stateKey][actionToString({0.4f, 0.0f})] = 50.0f;   // Prefer throttle
                qTable[stateKey][actionToString({0.0f, 0.0f})] = 0.0f;    // Neutral
                qTable[stateKey][actionToString({0.0f, 0.4f})] = -50.0f;  // Avoid brake
            } else if (error > 1.0f) { // Too fast
                qTable[stateKey][actionToString({0.0f, 0.4f})] = 50.0f;   // Prefer brake
                qTable[stateKey][actionToString({0.0f, 0.0f})] = 0.0f;    // Neutral
                qTable[stateKey][actionToString({0.4f, 0.0f})] = -50.0f;  // Avoid throttle
            } else { // Near target
                qTable[stateKey][actionToString({0.0f, 0.0f})] = 50.0f;   // Prefer neutral
                qTable[stateKey][actionToString({0.4f, 0.0f})] = 0.0f;    // Throttle sometimes
                qTable[stateKey][actionToString({0.0f, 0.4f})] = 0.0f;    // Brake sometimes
            }
        }
    }
    
    // Find action with highest Q-value
    std::string bestActionKey;
    float bestValue = -std::numeric_limits<float>::max();
    
    //printf("Q-values: ");
    for (const auto& actionPair : qTable[stateKey]) {
        //printf("[%s: %.1f] ", actionPair.first.c_str(), actionPair.second);
        if (actionPair.second > bestValue) {
            bestValue = actionPair.second;
            bestActionKey = actionPair.first;
        }
    }
    
    // If no best action found, default to no action
    if (bestActionKey.empty()) {
        printf("DEFAULT: No action found\n");
        return {0.0f, 0.0f};
    }
    
    // Parse best action
    std::vector<float> bestAction;
    std::istringstream iss(bestActionKey);
    std::string token;
    while (std::getline(iss, token, ',')) {
        bestAction.push_back(std::stof(token));
    }
    
    return bestAction;
}

int RLAgent::selectActionFromQTable(const std::string& stateKey) {
    std::string bestActionKey;
    float bestValue = -std::numeric_limits<float>::max();
    
    for (const auto& actionPair : qTable[stateKey]) {
        if (actionPair.second > bestValue) {
            bestValue = actionPair.second;
            bestActionKey = actionPair.first;
        }
    }
    
    // If we don't have any actions yet, return default
    if (bestActionKey.empty()) {
        return 2; // Default action index
    }
    
    // Parse best action string to match with action space
    std::vector<float> bestAction;
    std::istringstream iss(bestActionKey);
    std::string token;
    while (std::getline(iss, token, ',')) {
        bestAction.push_back(std::stof(token));
    }
    
    // Find matching action in action space
    for (size_t i = 0; i < actionSpace.size(); i++) {
        if (actionSpace[i][0] == bestAction[0] && actionSpace[i][1] == bestAction[1]) {
            return i;
        }
    }
    
    return 2; // Default action index if no match found
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

    // Use very high learning rate for dramatic updates
    float effectiveAlpha = 0.9f;

    // Apply clipping to prevent extreme Q-values
    float targetQ = reward + gamma * maxNextQ;
    targetQ = std::max(-1000.0f, std::min(1000.0f, targetQ));

    // Q-learning update with simulated experience replay
    float newQ = currentQ + effectiveAlpha * (targetQ - currentQ);
    qTable[stateKey][actionKey] = newQ;

    // Debug output for significant updates
    if (std::abs(newQ - currentQ) > 10.0f) {
        /* printf("Big Q update: %s, %s: %.1f -> %.1f (reward: %.1f)\n", 
            stateKey.c_str(), actionKey.c_str(), currentQ, newQ, reward); */
    }
}

void RLAgent::addExperience(const std::vector<float>& state, const std::vector<float>& action, 
    float reward, const std::vector<float>& nextState, bool done) {
    // Add experience to replay buffer
    if (replayBuffer.size() >= MAX_REPLAY_BUFFER_SIZE) {
        replayBuffer.pop_front();
    }

    replayBuffer.push_back({state, action, reward, nextState, done});
}

void RLAgent::trainFromReplay(int batchSize) {
    if (replayBuffer.size() < 5) return; // Need at least some experiences

    // Select random batch from replay buffer
    std::vector<int> indices(replayBuffer.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Train on mini-batch
    int actualBatchSize = std::min(batchSize, (int)replayBuffer.size());
    for (int i = 0; i < actualBatchSize; i++) {
        const Experience& exp = replayBuffer[indices[i]];
        updateQValues(exp.state, exp.action, exp.reward, exp.nextState);
    }

    // Also directly train on some synthetic experiences
    if (replayBuffer.size() >= 2) {
        // Add synthetic experience for being at target speed
        std::vector<float> targetState = {8.33f, 0.0f}; // At target speed
        std::vector<float> targetAction = {0.0f, 0.0f}; // Do nothing
        std::vector<float> targetNextState = {8.33f, 0.0f}; // Stay at target
        updateQValues(targetState, targetAction, 100.0f, targetNextState);

        // Add synthetic experience for being too slow
        std::vector<float> slowState = {4.0f, -4.33f}; // Below target speed
        std::vector<float> throttleAction = {0.4f, 0.0f}; // Throttle
        std::vector<float> speedingUpState = {6.0f, -2.33f}; // Getting closer
        updateQValues(slowState, throttleAction, 50.0f, speedingUpState);

        // Add synthetic experience for being too fast
        std::vector<float> fastState = {12.0f, 3.67f}; // Above target speed
        std::vector<float> brakeAction = {0.0f, 0.4f}; // Brake
        std::vector<float> slowingDownState = {10.0f, 1.67f}; // Getting closer
        updateQValues(fastState, brakeAction, 50.0f, slowingDownState);
    }
}

void RLAgent::decayExploration(float amount) {
    // Super aggressive decay
    static int steps = 0;
    steps++;

    // Very rapid decay over 50 steps
    epsilon = std::max(0.05f, initialEpsilon * (1.0f - steps / 50.0f));
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
    // DRASTICALLY simplified discretization - just 5 speed buckets and 5 error buckets
    std::vector<float> discretized;

    // Only care about speed and error
    if (state.size() > 0) {
        float speed = state[0];
        // Discretize to 5 buckets: very slow, slow, at target, fast, very fast
        if (speed < 4.0f) discretized.push_back(2.0f);
        else if (speed < 7.0f) discretized.push_back(6.0f);
        else if (speed < 10.0f) discretized.push_back(8.0f);
        else if (speed < 13.0f) discretized.push_back(12.0f);
        else discretized.push_back(16.0f);
    }

    if (state.size() > 1) {
        // Error: just 5 categories
        float error = state[1];
        if (error < -4.0f) discretized.push_back(-6.0f);
        else if (error < -1.0f) discretized.push_back(-2.0f);
        else if (error < 1.0f) discretized.push_back(0.0f);
        else if (error < 4.0f) discretized.push_back(2.0f);
        else discretized.push_back(6.0f);
    }

    return discretized;
}

void RLAgent::recordEpisodeMetrics(float reward, float qValue) {
    episodeRewards.push_back(reward);
    episodeQValues.push_back(qValue);
}

float RLAgent::getAverageEpisodeReward() const {
    if (episodeRewards.empty()) return 0.0f;
    
    float sum = std::accumulate(episodeRewards.begin(), episodeRewards.end(), 0.0f);
    return sum / episodeRewards.size();
}

float RLAgent::getAverageEpisodeQValue() const {
    if (episodeQValues.empty()) return 0.0f;
    
    float sum = std::accumulate(episodeQValues.begin(), episodeQValues.end(), 0.0f);
    return sum / episodeQValues.size();
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
    file.write(reinterpret_cast<const char*>(&initialEpsilon), sizeof(float));
    
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
    file.read(reinterpret_cast<char*>(&initialEpsilon), sizeof(float));
    
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

float RLAgent::getBestQValue() const {
    float best = -std::numeric_limits<float>::max();
    
    for (const auto& statePair : qTable) {
        for (const auto& actionPair : statePair.second) {
            best = std::max(best, actionPair.second);
        }
    }
    
    return qTable.empty() ? 0.0f : best;
}

float RLAgent::getWorstQValue() const {
    float worst = std::numeric_limits<float>::max();
    
    for (const auto& statePair : qTable) {
        for (const auto& actionPair : statePair.second) {
            worst = std::min(worst, actionPair.second);
        }
    }
    
    return qTable.empty() ? 0.0f : worst;
}

std::vector<float> RLAgent::getQDistribution(int bins) const {
    std::vector<float> distribution(bins, 0.0f);
    
    if (qTable.empty()) return distribution;
    
    // Find min and max Q-values
    float minQ = getWorstQValue();
    float maxQ = getBestQValue();
    
    // Avoid division by zero
    float range = maxQ - minQ;
    if (range <= 0.001f) return distribution;
    
    // Count Q-values in each bin
    for (const auto& statePair : qTable) {
        for (const auto& actionPair : statePair.second) {
            float q = actionPair.second;
            int bin = std::min(bins - 1, (int)((q - minQ) / range * bins));
            distribution[bin]++;
        }
    }
    
    // Normalize
    float total = std::accumulate(distribution.begin(), distribution.end(), 0.0f);
    if (total > 0.0f) {
        for (auto& val : distribution) {
            val /= total;
        }
    }
    
    return distribution;
}

// Enhanced Speed Control Task Implementation
EnhancedSpeedControlTask::EnhancedSpeedControlTask(float targetSpeed, float episodeDuration)
    : targetSpeed(targetSpeed),
      episodeDuration(episodeDuration),
      prevSpeed(0.0f),
      prevThrottle(0.0f),
      prevBrake(0.0f),
      speedErrorIntegral(0.0f) {
}

std::vector<float> EnhancedSpeedControlTask::getState(const Car& car, float deltaTime) {
    // Calculate acceleration (speed derivative)
    float acceleration = deltaTime > 0.001f ? (car.speed - prevSpeed) / deltaTime : 0.0f;
    
    // Update memory
    prevSpeed = car.speed;
    
    // Return enhanced state vector
    return {
        car.speed,                 // Current speed
        car.speed - targetSpeed,   // Error from target
        acceleration,              // Rate of change of speed
        prevThrottle,              // Previous throttle setting
        prevBrake                  // Previous brake setting
    };
}

float EnhancedSpeedControlTask::calculateReward(const Car& car, float deltaTime) {
    float speedError = car.speed - targetSpeed;
    float speedErrorAbs = std::abs(speedError);
    
    // Much stronger reward function - large rewards for being close to target
    float speedReward = 0.0f;
    
    // Very high reward when close to target (within 10%)
    if (speedErrorAbs < targetSpeed * 0.1f) {
        speedReward = 50.0f * (1.0f - speedErrorAbs / (targetSpeed * 0.1f));
    }
    // Medium reward when reasonably close (within 30%)
    else if (speedErrorAbs < targetSpeed * 0.3f) {
        speedReward = 20.0f * (1.0f - speedErrorAbs / (targetSpeed * 0.3f));
    }
    // Large penalty when far from target
    else {
        speedReward = -30.0f * speedErrorAbs / targetSpeed;
    }
    
    // Severe penalty for using throttle and brake simultaneously
    float controlPenalty = 0.0f;
    if (car.throttle > 0.1f && car.brake > 0.1f) {
        controlPenalty = -100.0f;
    }
    
    // Update state memory
    prevThrottle = car.throttle;
    prevBrake = car.brake;
    
    // Combined reward - much higher magnitude for faster learning
    return speedReward + controlPenalty;
}

bool EnhancedSpeedControlTask::isEpisodeComplete(float episodeTimer) const {
    return episodeTimer >= episodeDuration;
}

void EnhancedSpeedControlTask::resetEpisode(Car& car) {
    // Reset car to starting position with a small initial speed
    car.position = Vector3{ 0.0f, 0.5f, 0.0f };
    car.velocity = Vector3{ 0.0f, 0.0f, 0.0f };
    car.speed = 0.0f;
    car.rotation = 0.0f;
    car.throttle = 0.0f;
    car.brake = 0.0f;
    
    // Reset state memory
    prevSpeed = 0.0f;
    prevThrottle = 0.0f;
    prevBrake = 0.0f;
    speedErrorIntegral = 0.0f;
}

} // namespace CarGame