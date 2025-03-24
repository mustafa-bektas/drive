#pragma once

#include "raylib.h"
#include "dqn_environment.h"
#include "dqn_agent.h"
#include <vector>
#include <deque>
#include <string>

namespace CarGame {

class DQNVisualizer {
public:
    DQNVisualizer(int maxHistorySize = 300);
    
    void draw(const DQNEnvironment& env, const DQNAgent& agent);
    
    void recordEpisodeData(float reward, float speed, float targetSpeed);
    int startNewEpisode();
    
    void setLastAction(DQNEnvironment::Action action) { lastAction = action; }
    void setLastReward(float reward) { lastReward = reward; }
    
private:
    // Learning history
    struct EpisodeData {
        int episodeNumber;
        float totalReward;
    };
    
    std::vector<EpisodeData> episodeHistory;
    std::deque<float> rewardHistory;
    std::deque<float> speedHistory;
    std::deque<float> targetSpeedHistory;
    std::deque<float> lossHistory;
    int maxHistorySize;
    
    float currentEpisodeReward = 0.0f;
    int episodeCount = 0;
    
    // Current action and reward
    DQNEnvironment::Action lastAction = DQNEnvironment::Action::COAST;
    float lastReward = 0.0f;
    
    // Drawing methods
    void drawLearningProgress(int x, int y, int width, int height);
    void drawSpeedGraph(int x, int y, int width, int height, const DQNEnvironment& env);
    void drawAgentStats(int x, int y, const DQNAgent& agent);
    void drawControls(int x, int y, const DQNEnvironment& env);
    void drawLossGraph(int x, int y, int width, int height, const DQNAgent& agent);
    void drawActionExplanation(int x, int y, int width, int height);
    
    // Helper to get action name
    std::string getActionName(DQNEnvironment::Action action);
};

} // namespace CarGame