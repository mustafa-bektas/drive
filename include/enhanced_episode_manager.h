#pragma once

#include <string>
#include <vector>
#include <deque>
#include <fstream>
#include "car.h"
#include "rl_agent.h"

namespace CarGame {

class EnhancedEpisodeManager {
public:
    EnhancedEpisodeManager(float episodeDuration = 10.0f, float targetSpeed = 20.0f, 
                          int trainingStepsPerUpdate = 5, int experienceReplayBatchSize = 32);
    
    void update(float deltaTime, Car& car, RLAgent& agent);
    bool isEpisodeComplete() const;
    void resetEpisode(Car& car);
    void saveStats(const std::string& filename);
    void startNewEpisode(Car& car);
    
    // Getters
    int getEpisodeCount() const { return episodeCount; }
    float getCurrentReward() const { return currentReward; }
    float getCumulativeReward() const { return episodeCumulativeReward; }
    float getAverageReward() const { return episodeRewards.empty() ? 0.0f : averageReward; }
    float getBestReward() const { return bestReward; }
    float getEpisodeTimer() const { return episodeTimer; }
    float getTargetSpeed() const { return task.getTargetSpeed(); }
    float getSpeedError() const { return car ? car->speed - task.getTargetSpeed() : 0.0f; }
    
    // Visualization metrics
    int getSuccessfulEpisodes() const { return successfulEpisodes; }
    float getSuccessRate() const { 
        return episodeCount > 0 ? (float)successfulEpisodes / episodeCount : 0.0f; 
    }
    const std::deque<float>& getRewardHistory() const { return rewardHistory; }
    const std::deque<float>& getSpeedErrorHistory() const { return speedErrorHistory; }
    const std::deque<float>& getExplorationRateHistory() const { return explorationRateHistory; }
    
private:
    EnhancedSpeedControlTask task;
    float episodeDuration;
    float episodeTimer;
    int episodeCount;
    float currentReward;
    float episodeCumulativeReward;
    float averageReward;
    float bestReward;
    Car* car;  // Non-owning pointer
    
    // Training parameters
    int trainingStepsPerUpdate;
    int experienceReplayBatchSize;
    int framesSinceLastTraining;
    
    // Metrics
    std::vector<float> episodeRewards;
    std::vector<float> averageSpeedErrors;
    int successfulEpisodes;
    
    // Circular buffers for visualization
    std::deque<float> rewardHistory;
    std::deque<float> speedErrorHistory;
    std::deque<float> explorationRateHistory;
    const size_t MAX_HISTORY_SIZE = 1000;
    
    // Current episode state
    std::vector<float> currentState;
    std::vector<float> currentAction;
    bool firstFrame;
    
    void logEpisodeResults();
    bool isEpisodeSuccessful() const;
};

} // namespace CarGame