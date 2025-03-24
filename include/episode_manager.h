#pragma once

#include <string>
#include <vector>
#include <fstream>
#include "car.h"
#include "rl_agent.h"

namespace CarGame {

class EpisodeManager {
public:
    EpisodeManager(float episodeDuration = 30.0f, float targetSpeed = 20.0f);
    
    void update(float deltaTime, Car& car, RLAgent& agent);
    bool isEpisodeComplete() const;
    void resetEpisode(Car& car);
    void saveStats(const std::string& filename);
    
    // Getters
    int getEpisodeCount() const { return episodeCount; }
    float getCurrentReward() const { return currentReward; }
    float getAverageReward() const { return episodeRewards.empty() ? 0.0f : totalReward / episodeRewards.size(); }
    float getBestReward() const { return bestReward; }
    float getEpisodeTimer() const { return episodeTimer; }
    float getTargetSpeed() const { return targetSpeed; }
    float getSpeedError() const { return car ? car->speed - targetSpeed : 0.0f; }
    
private:
    float episodeDuration;
    float episodeTimer;
    float targetSpeed;
    int episodeCount;
    float currentReward;
    float totalReward;
    float bestReward;
    Car* car;  // Non-owning pointer
    
    std::vector<float> episodeRewards;
    std::vector<float> averageSpeedErrors;
    
    void calculateReward();
    void logEpisodeResults();
};
}
