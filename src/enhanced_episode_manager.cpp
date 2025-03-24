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
      trainingStepsPerUpdate(1), // Train every frame
      experienceReplayBatchSize(64), // Larger batch size
      framesSinceLastTraining(0),
      successfulEpisodes(0),
      firstFrame(true) {
}

void EnhancedEpisodeManager::update(float deltaTime, Car& car, RLAgent& agent) {
    this->car = &car;
    episodeTimer += deltaTime;
    
    // Simplified state: just speed and error to target speed
    std::vector<float> nextState = {
        car.speed,
        car.speed - task.getTargetSpeed()
    };
    
    // On first frame of episode, just store the initial state
    if (firstFrame) {
        currentState = nextState;
        currentAction = agent.getAction(currentState);
        firstFrame = false;
        return;
    }
    
    // Calculate reward - COMPLETELY SIMPLIFIED ULTRA CLEAR REWARD
    float speedError = std::abs(car.speed - task.getTargetSpeed());
    float targetSpeed = task.getTargetSpeed();
    
    // BINARY SUCCESS/FAILURE REWARD
    if (speedError < targetSpeed * 0.1f) { // Within 10% of target
        currentReward = 100.0f; // BIG REWARD for being on target
    } 
    else if (speedError < targetSpeed * 0.2f) { // Within 20% of target
        currentReward = 50.0f; // Moderate reward for being close
    }
    else if (speedError < targetSpeed * 0.5f) { // Within 50% of target
        currentReward = 20.0f - speedError * 2.0f; // Small reward for being in right direction
    }
    else {
        // Large penalty growing with distance
        currentReward = -30.0f - speedError * 5.0f;
    }
    
    // Severe penalty for using throttle and brake simultaneously
    if (car.throttle > 0.1f && car.brake > 0.1f) {
        currentReward -= 100.0f;
    }
    
    // Update cumulative reward
    episodeCumulativeReward += currentReward;
    
    // Store experience in replay buffer
    bool done = isEpisodeComplete();
    agent.addExperience(currentState, currentAction, currentReward, nextState, done);
    
    // Record metrics for visualization
    agent.recordEpisodeMetrics(currentReward, 0.0f);
    
    // Update visualization histories
    rewardHistory.push_back(currentReward);
    if (rewardHistory.size() > MAX_HISTORY_SIZE) rewardHistory.pop_front();
    
    speedErrorHistory.push_back(car.speed - task.getTargetSpeed());
    if (speedErrorHistory.size() > MAX_HISTORY_SIZE) speedErrorHistory.pop_front();
    
    explorationRateHistory.push_back(agent.getExplorationRate());
    if (explorationRateHistory.size() > MAX_HISTORY_SIZE) explorationRateHistory.pop_front();
    
    // Train on EVERY frame for much faster learning
    for (int i = 0; i < 20; i++) {  // 20 training steps every frame!
        agent.trainFromReplay(experienceReplayBatchSize);
    }
    
    // Decay exploration aggressively
    agent.decayExploration();
    
    // Set next action
    currentState = nextState;
    currentAction = agent.getAction(currentState);
    
    // Apply action to car
    car.throttle = currentAction[0];
    car.brake = currentAction[1];
    
    // Print current car state
    printf("CAR - Speed: %.1f km/h, Throttle: %.2f, Brake: %.2f\n", 
           car.speed * 3.6f, car.throttle, car.brake);
    
    // Check if episode is complete - shorter episodes
    if (isEpisodeComplete()) {
        logEpisodeResults();
        
        if (isEpisodeSuccessful()) {
            successfulEpisodes++;
            // Print success message for visibility
            printf("*** EPISODE %d SUCCEEDED! Good speed control! ***\n", episodeCount);
        }
        else {
            printf("Episode %d failed. Average speed error: %.2f\n", 
                   episodeCount, std::abs(car.speed - task.getTargetSpeed()));
        }
        
        // Reset for next episode
        startNewEpisode(car);
    }
}

bool EnhancedEpisodeManager::isEpisodeComplete() const {
    return task.isEpisodeComplete(episodeTimer);
}

void EnhancedEpisodeManager::resetEpisode(Car& car) {
    // Reset car to starting position
    car.position = Vector3{ 0.0f, 0.5f, 0.0f };
    car.velocity = Vector3{ 0.0f, 0.0f, 0.0f };
    car.speed = 0.0f;
    car.rotation = 0.0f;
    car.throttle = 0.0f;
    car.brake = 0.0f;
    
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
}

bool EnhancedEpisodeManager::isEpisodeSuccessful() const {
    // Consider episode successful if final speed error is within 10% of target
    if (!car) return false;
    
    float targetSpeed = task.getTargetSpeed();
    float finalSpeedError = std::abs(car->speed - targetSpeed);
    float errorThreshold = targetSpeed * 0.1f; // 10% of target speed
    
    return finalSpeedError < errorThreshold;
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