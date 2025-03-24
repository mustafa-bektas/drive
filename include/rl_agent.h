#pragma once

#include <vector>
#include <map>
#include <string>
#include <random>
#include "car.h"

namespace CarGame {

class RLAgent {
public:
    RLAgent(float learningRate = 0.1, float discountFactor = 0.9, float explorationRate = 0.3);
    
    std::vector<float> getAction(const std::vector<float>& state);
    void updateQValues(const std::vector<float>& state, const std::vector<float>& action, 
                      float reward, const std::vector<float>& nextState);
    
    // Training utilities
    void saveModel(const std::string& filename);
    bool loadModel(const std::string& filename);
    void setExplorationRate(float rate) { epsilon = rate; }
    float getExplorationRate() const { return epsilon; }
    
    // Visualization
    int getQTableSize() const { return qTable.size(); }
    float getAverageQValue() const;
    
private:
    float alpha;      // Learning rate
    float gamma;      // Discount factor
    float epsilon;    // Exploration rate
    
    // Simple Q-table for discrete states/actions
    std::map<std::string, std::map<std::string, float>> qTable;
    
    std::string stateToString(const std::vector<float>& state);
    std::string actionToString(const std::vector<float>& action);
    std::vector<float> discretizeState(const std::vector<float>& state);
    std::random_device rd;
    std::mt19937 gen;
};

// RL task interface
class RLTask {
public:
    virtual std::vector<float> getState(const Car& car) = 0;
    virtual float calculateReward(const Car& car) = 0;
    virtual bool isEpisodeComplete(const Car& car) = 0;
    virtual void resetEpisode(Car& car) = 0;
    virtual ~RLTask() = default;
};

// Speed control task
class SpeedControlTask : public RLTask {
public:
    SpeedControlTask(float targetSpeed = 20.0f);
    
    std::vector<float> getState(const Car& car) override;
    float calculateReward(const Car& car) override;
    bool isEpisodeComplete(const Car& car) override;
    void resetEpisode(Car& car) override;
    
private:
    float targetSpeed;
    float episodeDuration;
    float episodeTimer;
};

} // namespace CarGame