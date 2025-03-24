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
    
    // Simplified action space - just 5 actions for faster learning
    actionSpace = {
        {0.0f, 0.0f},   // No throttle, no brake
        {0.5f, 0.0f},   // Medium throttle
        {1.0f, 0.0f},   // Full throttle
        {0.0f, 0.5f},   // Medium brake
    };
    
    // Initialize Q-table with optimistic values to encourage exploration
    for (const auto& action : actionSpace) {
        std::string actionKey = actionToString(action);
        qTable["0;0"][actionKey] = 10.0f;  // Optimistic initialization
    }
}

std::vector<float> RLAgent::getAction(const std::vector<float>& state) {
    // Discretize the state for table lookup
    std::vector<float> discreteState = discretizeState(state);
    std::string stateKey = stateToString(discreteState);
    
    // Exploration: random action with probability epsilon
    std::uniform_real_distribution<> distr(0.0, 1.0);
    if (distr(gen) < epsilon) {
        // Return random action from action space
        std::uniform_int_distribution<> actionDistr(0, actionSpace.size() - 1);
        return actionSpace[actionDistr(gen)];
    }
    
    // Exploitation: best known action
    if (qTable.find(stateKey) == qTable.end()) {
        // State not seen before, initialize with zero values
        qTable[stateKey] = std::map<std::string, float>();
    }
    
    // If no actions exist yet, use a default action
    if (qTable[stateKey].empty()) {
        return actionSpace[2]; // Default: 50% throttle, no brake
    }
    
    // Find action with highest Q-value
    int actionIndex = selectActionFromQTable(stateKey);
    return actionSpace[actionIndex];
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
    
    // Q-learning update
    float newQ = currentQ + alpha * (reward + gamma * maxNextQ - currentQ);
    qTable[stateKey][actionKey] = newQ;
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
    if (replayBuffer.size() < batchSize) return;
    
    // Select random batch from replay buffer
    std::vector<int> indices(replayBuffer.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Train on mini-batch
    for (int i = 0; i < std::min(batchSize, (int)replayBuffer.size()); i++) {
        const Experience& exp = replayBuffer[indices[i]];
        updateQValues(exp.state, exp.action, exp.reward, exp.nextState);
    }
}

void RLAgent::decayExploration(float amount) {
    // Much more aggressive linear decay
    static int steps = 0;
    steps++;
    
    // Rapid decay over just 100 steps
    epsilon = std::max(0.05f, initialEpsilon * (1.0f - steps / 100.0f));
    
    // Reset exploration periodically to escape local optima
    if (steps % 200 == 0) {
        epsilon = std::min(0.5f, epsilon + 0.2f);
        printf("Exploration bump! New epsilon: %.2f\n", epsilon);
    }
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
    // Much coarser discretization for faster learning
    std::vector<float> discretized;
    
    // Only use first two state components (speed and error) for simplicity
    if (state.size() > 0) {
        // Speed: discretize to nearest 2.0 m/s
        float speed = state[0];
        discretized.push_back(std::round(speed / 2.0f) * 2.0f);
    }
    
    if (state.size() > 1) {
        // Error: discretize to just 5 possible values (-large, -small, zero, small, large)
        float error = state[1];
        if (error < -4.0f) discretized.push_back(-5.0f);
        else if (error < -1.0f) discretized.push_back(-2.0f);
        else if (error < 1.0f) discretized.push_back(0.0f);
        else if (error < 4.0f) discretized.push_back(2.0f);
        else discretized.push_back(5.0f);
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