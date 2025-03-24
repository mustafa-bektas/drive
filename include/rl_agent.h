#pragma once

#include <vector>
#include <map>
#include <string>
#include <random>
#include <deque>
#include "car.h"

namespace CarGame {

// Experience structure for replay buffer
struct Experience {
    std::vector<float> state;
    std::vector<float> action;
    float reward;
    std::vector<float> nextState;
    bool done;
};

class RLAgent {
public:
    RLAgent(float learningRate = 0.2, float discountFactor = 0.95, float explorationRate = 1.0);
    
    std::vector<float> getAction(const std::vector<float>& state);
    void updateQValues(const std::vector<float>& state, const std::vector<float>& action, 
                      float reward, const std::vector<float>& nextState);
    
    // Experience replay
    void addExperience(const std::vector<float>& state, const std::vector<float>& action, 
                      float reward, const std::vector<float>& nextState, bool done);
    void trainFromReplay(int batchSize = 32);
    
    // Training utilities
    void saveModel(const std::string& filename);
    bool loadModel(const std::string& filename);
    void setExplorationRate(float rate) { epsilon = rate; }
    float getExplorationRate() const { return epsilon; }
    void decayExploration(float amount = 0.005f);
    
    // Visualization
    int getQTableSize() const { return qTable.size(); }
    float getAverageQValue() const;
    float getBestQValue() const;
    float getWorstQValue() const;
    std::vector<float> getQDistribution(int bins = 10) const;
    const std::deque<float>& getLearningCurve() const { return learningCurve; }
    
    // Action space management
    const std::vector<std::vector<float>>& getActionSpace() const { return actionSpace; }
    int getActionSpaceSize() const { return actionSpace.size(); }
    
    // Reset learning metrics for new episode
    void resetEpisodeMetrics() { episodeRewards.clear(); episodeQValues.clear(); }
    void recordEpisodeMetrics(float reward, float qValue);
    float getAverageEpisodeReward() const;
    float getAverageEpisodeQValue() const;
    
private:
    float alpha;         // Learning rate
    float gamma;         // Discount factor
    float epsilon;       // Exploration rate
    float initialEpsilon; // Starting exploration rate
    
    // Simple Q-table for discrete states/actions
    std::map<std::string, std::map<std::string, float>> qTable;
    
    // Action space - discrete set of throttle/brake combinations
    std::vector<std::vector<float>> actionSpace;
    
    // Experience replay buffer
    std::deque<Experience> replayBuffer;
    const size_t MAX_REPLAY_BUFFER_SIZE = 10000;
    
    // Learning metrics
    std::deque<float> learningCurve;  // Tracks average reward per episode
    const size_t MAX_LEARNING_CURVE_SIZE = 100;
    
    // Episode metrics
    std::vector<float> episodeRewards;
    std::vector<float> episodeQValues;
    
    // Utility methods
    std::string stateToString(const std::vector<float>& state);
    std::string actionToString(const std::vector<float>& action);
    std::vector<float> discretizeState(const std::vector<float>& state);
    int selectActionFromQTable(const std::string& stateKey);
    
    std::random_device rd;
    std::mt19937 gen;
};

// Enhanced speed control task with better state representation
class EnhancedSpeedControlTask {
public:
    EnhancedSpeedControlTask(float targetSpeed = 20.0f, float episodeDuration = 30.0f);
    
    std::vector<float> getState(const Car& car, float deltaTime);
    float calculateReward(const Car& car, float deltaTime);
    bool isEpisodeComplete(float episodeTimer) const;
    void resetEpisode(Car& car);
    float getTargetSpeed() const { return targetSpeed; }
    
private:
    float targetSpeed;
    float episodeDuration;
    
    // State memory for derivatives
    float prevSpeed;
    float prevThrottle;
    float prevBrake;
    float speedErrorIntegral;
    
    // Constants for reward calculation
    const float SPEED_ERROR_WEIGHT = 10.0f;
    const float SMOOTHNESS_WEIGHT = 5.0f;
    const float CONTROL_PENALTY = 20.0f;
    const float THROTTLE_CHANGE_PENALTY = 3.0f;
};

} // namespace CarGame