#include "../include/episode_manager.h"

namespace CarGame {
EpisodeManager::EpisodeManager(float episodeDuration, float targetSpeed)
    : episodeDuration(episodeDuration), 
      episodeTimer(0.0f),
      targetSpeed(targetSpeed),
      episodeCount(0),
      currentReward(0.0f),
      totalReward(0.0f),
      bestReward(-std::numeric_limits<float>::max()),
      car(nullptr) {
}

void EpisodeManager::update(float deltaTime, Car& car, RLAgent& agent) {
    this->car = &car;
    episodeTimer += deltaTime;
    
    // Update reward calculation
    calculateReward();
    
    // Gradually decrease exploration rate over time
    if (episodeCount > 10) {
        float newEpsilon = std::max(0.05f, agent.getExplorationRate() - 0.001f);
        agent.setExplorationRate(newEpsilon);
    }
    
    // Check if episode is complete
    if (isEpisodeComplete()) {
        logEpisodeResults();
        resetEpisode(car);
    }
}

bool EpisodeManager::isEpisodeComplete() const {
    return episodeTimer >= episodeDuration;
}

void EpisodeManager::resetEpisode(Car& car) {
    // Reset car to starting position
    car.position = Vector3{ 0.0f, 0.5f, 0.0f };
    car.velocity = Vector3{ 0.0f, 0.0f, 0.0f };
    car.speed = 0.0f;
    car.rotation = 0.0f;
    car.throttle = 0.0f;
    car.brake = 0.0f;
    
    // Reset episode timer
    episodeTimer = 0.0f;
    episodeCount++;
}

void EpisodeManager::calculateReward() {
    if (!car) return;
    
    // Simple reward: negative error from target speed
    float speedError = car->speed - targetSpeed;
    currentReward = -std::exp(-speedError) - 1;

    if (car->throttle > 0.1f & car->brake> 0.1f) {
        currentReward -= 10.0f;
    }
}

void EpisodeManager::logEpisodeResults() {
    // Calculate average reward for this episode
    float episodeAvgReward = currentReward;
    float avgSpeedError = std::abs(car->speed - targetSpeed);
    
    // Store episode stats
    episodeRewards.push_back(episodeAvgReward);
    averageSpeedErrors.push_back(avgSpeedError);
    
    // Update totals
    totalReward += episodeAvgReward;
    bestReward = std::max(bestReward, episodeAvgReward);
    
    // Log to console
    printf("Episode %d complete. Reward: %.2f, Avg Speed Error: %.2f\n", 
           episodeCount, episodeAvgReward, avgSpeedError);
}

void EpisodeManager::saveStats(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }
    
    file << "Episode,Reward,SpeedError\n";
    for (size_t i = 0; i < episodeRewards.size(); ++i) {
        file << i + 1 << "," << episodeRewards[i] << "," << averageSpeedErrors[i] << "\n";
    }
    
    file.close();
}
} // namespace CarGame