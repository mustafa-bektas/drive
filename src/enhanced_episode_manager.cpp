#include "../include/enhanced_episode_manager.h"
#include <algorithm>
#include <numeric>

namespace CarGame {

EnhancedEpisodeManager::EnhancedEpisodeManager(float episodeDuration, float targetSpeed, 
                                             int trainingStepsPerUpdate, int experienceReplayBatchSize)
    : task(targetSpeed, episodeDuration),
      episodeDuration(episodeDuration), 
      episodeTimer(0.0f),
      episodeCount(0),
      currentReward(0.0f),
      episodeCumulativeReward(0.0f),
      averageReward(0.0f),
      bestReward(-std::numeric_limits<float>::max()),
      car(nullptr),
      trainingStepsPerUpdate(trainingStepsPerUpdate),
      experienceReplayBatchSize(experienceReplayBatchSize),
      framesSinceLastTraining(0),
      successfulEpisodes(0),
      firstFrame(true) {
}

void EnhancedEpisodeManager::update(float deltaTime, Car& car, RLAgent& agent) {
    this->car = &car;
    episodeTimer += deltaTime;
    
    // Get current state from task
    std::vector<float> nextState = task.getState(car, deltaTime);
    
    // On first frame of episode, just store the initial state
    if (firstFrame) {
        currentState = nextState;
        currentAction = agent.getAction(currentState);
        firstFrame = false;
        return;
    }
    
    // Calculate reward
    currentReward = task.calculateReward(car, deltaTime);
    episodeCumulativeReward += currentReward;
    
    // Store experience in replay buffer
    bool done = isEpisodeComplete();
    agent.addExperience(currentState, currentAction, currentReward, nextState, done);
    
    // Record metrics for visualization
    agent.recordEpisodeMetrics(currentReward, 0.0f); // Could add actual Q-value here
    
    // Update visualization histories
    rewardHistory.push_back(currentReward);
    if (rewardHistory.size() > MAX_HISTORY_SIZE) rewardHistory.pop_front();
    
    speedErrorHistory.push_back(car.speed - task.getTargetSpeed());
    if (speedErrorHistory.size() > MAX_HISTORY_SIZE) speedErrorHistory.pop_front();
    
    explorationRateHistory.push_back(agent.getExplorationRate());
    if (explorationRateHistory.size() > MAX_HISTORY_SIZE) explorationRateHistory.pop_front();
    
    // Train using experience replay at regular intervals
    framesSinceLastTraining++;
    if (framesSinceLastTraining >= trainingStepsPerUpdate) {
        for (int i = 0; i < 5; i++) {  // Multiple training steps per update
            agent.trainFromReplay(experienceReplayBatchSize);
        }
        framesSinceLastTraining = 0;
        
        // Decay exploration rate
        agent.decayExploration();
    }
    
    // Set next action
    currentState = nextState;
    currentAction = agent.getAction(currentState);
    
    // Apply action to car
    car.throttle = currentAction[0];
    car.brake = currentAction[1];
    
    // Check if episode is complete
    if (isEpisodeComplete()) {
        logEpisodeResults();
        
        // Check if episode was successful
        if (isEpisodeSuccessful()) {
            successfulEpisodes++;
        }
        
        // Reset for next episode
        startNewEpisode(car);
    }
}

bool EnhancedEpisodeManager::isEpisodeComplete() const {
    return task.isEpisodeComplete(episodeTimer);
}

void EnhancedEpisodeManager::resetEpisode(Car& car) {
    // Reset task and car
    task.resetEpisode(car);
    
    // Reset episode timer
    episodeTimer = 0.0f;
    episodeCumulativeReward = 0.0f;
    firstFrame = true;
}

void EnhancedEpisodeManager::startNewEpisode(Car& car) {
    resetEpisode(car);
    episodeCount++;
}

void EnhancedEpisodeManager::logEpisodeResults() {
    // Store episode stats
    episodeRewards.push_back(episodeCumulativeReward);
    
    float averageSpeedError = 0.0f;
    if (!speedErrorHistory.empty()) {
        averageSpeedError = std::accumulate(speedErrorHistory.begin(), speedErrorHistory.end(), 0.0f) / 
                           speedErrorHistory.size();
    }
    averageSpeedErrors.push_back(std::abs(averageSpeedError));
    
    // Update totals
    if (!episodeRewards.empty()) {
        float sum = std::accumulate(episodeRewards.begin(), episodeRewards.end(), 0.0f);
        averageReward = sum / episodeRewards.size();
    }
    
    bestReward = std::max(bestReward, episodeCumulativeReward);
    
    // Log to console
    printf("Episode %d complete. Reward: %.2f, Avg Speed Error: %.2f, Success: %s\n", 
           episodeCount, episodeCumulativeReward, std::abs(averageSpeedError),
           isEpisodeSuccessful() ? "YES" : "NO");
}

bool EnhancedEpisodeManager::isEpisodeSuccessful() const {
    // Consider episode successful if average speed error is within 10% of target
    float averageSpeedError = 0.0f;
    if (!speedErrorHistory.empty()) {
        averageSpeedError = std::accumulate(speedErrorHistory.begin(), speedErrorHistory.end(), 0.0f) / 
                           speedErrorHistory.size();
    }
    
    float targetSpeed = task.getTargetSpeed();
    float errorThreshold = targetSpeed * 0.1f; // 10% of target speed
    
    return std::abs(averageSpeedError) < errorThreshold;
}

void EnhancedEpisodeManager::saveStats(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }
    
    file << "Episode,Reward,SpeedError,Success\n";
    for (size_t i = 0; i < episodeRewards.size(); ++i) {
        bool success = i < episodeRewards.size() - 1 ? 
                      std::abs(averageSpeedErrors[i]) < task.getTargetSpeed() * 0.1f : 
                      isEpisodeSuccessful();
                      
        file << i + 1 << "," << episodeRewards[i] << "," << averageSpeedErrors[i] 
             << "," << (success ? 1 : 0) << "\n";
    }
    
    file.close();
}

} // namespace CarGame