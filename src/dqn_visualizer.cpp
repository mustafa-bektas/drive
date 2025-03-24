#include "dqn_visualizer.h"
#include <algorithm>
#include <cmath>

namespace CarGame {

DQNVisualizer::DQNVisualizer(int maxHistorySize)
    : maxHistorySize(maxHistorySize) {
}

void DQNVisualizer::draw(const DQNEnvironment& env, const DQNAgent& agent) {
    // Draw learning visualizations
    int screenWidth = GetScreenWidth();
    int screenHeight = GetScreenHeight();
    
    // Draw speed graph
    drawSpeedGraph(20, 20, 500, 150, env);
    
    // Draw learning progress chart
    drawLearningProgress(20, 190, 500, 150);
    
    // Draw loss graph
    drawLossGraph(20, 360, 500, 150, agent);
    
    // Draw agent statistics
    drawAgentStats(20, 530, agent);
    
    // Draw current controls
    drawControls(20, 600, env);
    
    // Draw action explanation panel
    drawActionExplanation(540, 20, 300, 300);
}

void DQNVisualizer::recordEpisodeData(float reward, float speed, float targetSpeed) {
    currentEpisodeReward += reward;
    
    // Record speed for history graph
    speedHistory.push_back(speed);
    if (speedHistory.size() > maxHistorySize) {
        speedHistory.pop_front();
    }
    
    // Record target speed
    targetSpeedHistory.push_back(targetSpeed);
    if (targetSpeedHistory.size() > maxHistorySize) {
        targetSpeedHistory.pop_front();
    }
}

int DQNVisualizer::startNewEpisode() {
    episodeHistory.push_back({episodeCount, currentEpisodeReward});
    
    // Record reward for history graph
    rewardHistory.push_back(currentEpisodeReward);
    if (rewardHistory.size() > maxHistorySize) {
        rewardHistory.pop_front();
    }
    
    // Reset episode counters
    currentEpisodeReward = 0.0f;
    episodeCount++;
    return episodeCount;
}

void DQNVisualizer::drawLearningProgress(int x, int y, int width, int height) {
    // Draw panel background
    DrawRectangle(x, y, width, height, {200, 200, 200, 180});
    
    // Draw title
    DrawText("Learning Progress (Episode Rewards)", x + 10, y + 10, 20, DARKBLUE);
    
    // Draw reward history graph
    if (!rewardHistory.empty()) {
        int graphX = x + 10;
        int graphY = y + 40;
        int graphWidth = width - 20;
        int graphHeight = height - 50;
        
        // Draw axes
        DrawLine(graphX, graphY + graphHeight, graphX + graphWidth, graphY + graphHeight, BLACK);
        DrawLine(graphX, graphY, graphX, graphY + graphHeight, BLACK);
        
        // Find min and max rewards for scaling
        float minReward = *std::min_element(rewardHistory.begin(), rewardHistory.end());
        float maxReward = *std::max_element(rewardHistory.begin(), rewardHistory.end());
        minReward = std::min(0.0f, minReward);  // Always include zero
        maxReward = std::max(10.0f, maxReward);  // At least go to 10.0
        
        float yScale = graphHeight / (maxReward - minReward);
        float xScale = static_cast<float>(graphWidth) / std::max(1, static_cast<int>(rewardHistory.size() - 1));
        
        // Draw graph lines
        for (size_t i = 0; i < rewardHistory.size() - 1; i++) {
            float x1 = graphX + i * xScale;
            float y1 = graphY + graphHeight - (rewardHistory[i] - minReward) * yScale;
            float x2 = graphX + (i + 1) * xScale;
            float y2 = graphY + graphHeight - (rewardHistory[i + 1] - minReward) * yScale;
            
            DrawLine(x1, y1, x2, y2, RED);
        }
        
        // Draw min/max/current labels
        DrawText(TextFormat("Max: %.2f", maxReward), graphX + 10, graphY + 5, 16, BLACK);
        DrawText(TextFormat("Min: %.2f", minReward), graphX + 10, graphY + graphHeight - 20, 16, BLACK);
        
        if (!rewardHistory.empty()) {
            DrawText(TextFormat("Latest: %.2f", rewardHistory.back()), 
                     graphX + graphWidth - 120, graphY + 5, 16, BLACK);
        }
    } else {
        DrawText("Waiting for episode data...", x + 20, y + 60, 20, GRAY);
    }
}

void DQNVisualizer::drawSpeedGraph(int x, int y, int width, int height, const DQNEnvironment& env) {
    // Draw panel background
    DrawRectangle(x, y, width, height, {200, 200, 200, 180});
    
    // Draw title
    DrawText("Speed Control Performance", x + 10, y + 10, 20, DARKBLUE);
    
    // Draw current and target speed
    float targetSpeed = env.getTargetSpeed() * 3.6f;  // Convert to km/h
    float currentSpeed = env.getCar().speed * 3.6f;   // Convert to km/h
    
    DrawText(TextFormat("Target: %.1f km/h", targetSpeed), x + 10, y + 40, 20, DARKGREEN);
    DrawText(TextFormat("Current: %.1f km/h", currentSpeed), x + 200, y + 40, 20, RED);
    DrawText(TextFormat("Last Reward: %.2f", lastReward), x + 350, y + 40, 20, BLUE);
    
    // Draw speed history graph
    if (!speedHistory.empty() && !targetSpeedHistory.empty()) {
        int graphX = x + 10;
        int graphY = y + 70;
        int graphWidth = width - 20;
        int graphHeight = height - 80;
        
        // Draw axes
        DrawLine(graphX, graphY + graphHeight, graphX + graphWidth, graphY + graphHeight, BLACK);
        DrawLine(graphX, graphY, graphX, graphY + graphHeight, BLACK);
        
        // Fixed scale from 0 to max speed + buffer
        float maxDisplaySpeed = targetSpeed * 1.5f;
        float yScale = graphHeight / maxDisplaySpeed;
        float xScale = static_cast<float>(graphWidth) / std::max(1, static_cast<int>(speedHistory.size() - 1));
        
        // Draw target speed line
        for (size_t i = 0; i < targetSpeedHistory.size() - 1; i++) {
            float x1 = graphX + i * xScale;
            float y1 = graphY + graphHeight - (targetSpeedHistory[i] * 3.6f) * yScale;
            float x2 = graphX + (i + 1) * xScale;
            float y2 = graphY + graphHeight - (targetSpeedHistory[i + 1] * 3.6f) * yScale;
            
            DrawLine(x1, y1, x2, y2, {0, 100, 0, 128});
        }
        
        // Draw current speed line
        for (size_t i = 0; i < speedHistory.size() - 1; i++) {
            float x1 = graphX + i * xScale;
            float y1 = graphY + graphHeight - (speedHistory[i] * 3.6f) * yScale;
            float x2 = graphX + (i + 1) * xScale;
            float y2 = graphY + graphHeight - (speedHistory[i + 1] * 3.6f) * yScale;
            
            DrawLine(x1, y1, x2, y2, RED);
        }
        
        // Draw scale
        DrawText("0", graphX - 20, graphY + graphHeight - 10, 16, BLACK);
        DrawText(TextFormat("%.0f", maxDisplaySpeed), graphX - 40, graphY, 16, BLACK);
    } else {
        DrawText("Waiting for speed data...", x + 20, y + 90, 20, GRAY);
    }
}

void DQNVisualizer::drawLossGraph(int x, int y, int width, int height, const DQNAgent& agent) {
    // Draw panel background
    DrawRectangle(x, y, width, height, {200, 200, 200, 180});
    
    // Draw title
    DrawText("Training Loss", x + 10, y + 10, 20, DARKBLUE);
    
    // Get loss history
    const auto& agentLossHistory = agent.getLossHistory();
    
    // Copy loss to our history if needed
    if (!agentLossHistory.empty() && (lossHistory.empty() || lossHistory.back() != agentLossHistory.back())) {
        lossHistory.push_back(agentLossHistory.back());
        if (lossHistory.size() > maxHistorySize) {
            lossHistory.pop_front();
        }
    }
    
    // Draw loss history graph
    if (!lossHistory.empty()) {
        int graphX = x + 10;
        int graphY = y + 40;
        int graphWidth = width - 20;
        int graphHeight = height - 50;
        
        // Draw axes
        DrawLine(graphX, graphY + graphHeight, graphX + graphWidth, graphY + graphHeight, BLACK);
        DrawLine(graphX, graphY, graphX, graphY + graphHeight, BLACK);
        
        // Find min and max loss for scaling
        float minLoss = 0.0f;
        float maxLoss = std::max(1.0f, *std::max_element(lossHistory.begin(), lossHistory.end()));
        
        float yScale = graphHeight / (maxLoss - minLoss);
        float xScale = static_cast<float>(graphWidth) / std::max(1, static_cast<int>(lossHistory.size() - 1));
        
        // Draw graph lines
        for (size_t i = 0; i < lossHistory.size() - 1; i++) {
            float x1 = graphX + i * xScale;
            float y1 = graphY + graphHeight - (lossHistory[i] - minLoss) * yScale;
            float x2 = graphX + (i + 1) * xScale;
            float y2 = graphY + graphHeight - (lossHistory[i + 1] - minLoss) * yScale;
            
            DrawLine(x1, y1, x2, y2, ORANGE);
        }
        
        // Draw current loss
        if (!lossHistory.empty()) {
            DrawText(TextFormat("Current Loss: %.4f", lossHistory.back()), 
                     graphX + 10, graphY + 5, 16, BLACK);
        }
    } else {
        DrawText("Waiting for loss data...", x + 20, y + 60, 20, GRAY);
    }
}

void DQNVisualizer::drawAgentStats(int x, int y, const DQNAgent& agent) {
    // Draw panel background
    DrawRectangle(x, y, 500, 60, {200, 200, 200, 180});
    
    // Draw title
    DrawText("Agent Statistics", x + 10, y + 10, 20, DARKBLUE);
    
    // Draw statistics
    DrawText(TextFormat("Episodes: %d", episodeCount), x + 20, y + 35, 18, BLACK);
    DrawText(TextFormat("Exploration Rate: %.2f", agent.getExplorationRate()), x + 180, y + 35, 18, BLACK);
    DrawText(TextFormat("Updates: %d", agent.getUpdateCount()), x + 380, y + 35, 18, BLACK);
}

void DQNVisualizer::drawControls(int x, int y, const DQNEnvironment& env) {
    // Draw panel background
    DrawRectangle(x, y, 500, 100, {200, 200, 200, 180});
    
    // Draw title
    DrawText("Vehicle Controls", x + 10, y + 10, 20, DARKBLUE);
    
    // Draw control values
    const Car& car = env.getCar();
    
    // Draw throttle bar
    DrawText("Throttle:", x + 20, y + 40, 18, BLACK);
    DrawRectangle(x + 100, y + 40, 150, 20, LIGHTGRAY);
    DrawRectangle(x + 100, y + 40, car.throttle * 150, 20, GREEN);
    DrawText(TextFormat("%.2f", car.throttle), x + 260, y + 40, 18, BLACK);
    
    // Draw brake bar
    DrawText("Brake:", x + 20, y + 70, 18, BLACK);
    DrawRectangle(x + 100, y + 70, 150, 20, LIGHTGRAY);
    DrawRectangle(x + 100, y + 70, car.brake * 150, 20, RED);
    DrawText(TextFormat("%.2f", car.brake), x + 260, y + 70, 18, BLACK);
    
    // Draw current action
    DrawText(TextFormat("Current Action: %s", getActionName(lastAction).c_str()), 
              x + 320, y + 55, 18, DARKBLUE);
}

void DQNVisualizer::drawActionExplanation(int x, int y, int width, int height) {
    // Draw panel background
    DrawRectangle(x, y, width, height, {200, 200, 200, 180});
    
    // Draw title
    DrawText("DQN Action Space", x + 10, y + 10, 20, DARKBLUE);
    
    // List all actions
    const int lineHeight = 25;
    int textY = y + 40;
    
    DrawText("0: STRONG_BRAKE    (Brake 1.0)", x + 20, textY, 16, BLACK); textY += lineHeight;
    DrawText("1: MEDIUM_BRAKE    (Brake 0.66)", x + 20, textY, 16, BLACK); textY += lineHeight;
    DrawText("2: LIGHT_BRAKE     (Brake 0.33)", x + 20, textY, 16, BLACK); textY += lineHeight;
    DrawText("3: COAST           (No input)", x + 20, textY, 16, BLACK); textY += lineHeight;
    DrawText("4: LIGHT_THROTTLE  (Throttle 0.25)", x + 20, textY, 16, BLACK); textY += lineHeight;
    DrawText("5: MEDIUM_THROTTLE (Throttle 0.5)", x + 20, textY, 16, BLACK); textY += lineHeight;
    DrawText("6: STRONG_THROTTLE (Throttle 0.75)", x + 20, textY, 16, BLACK); textY += lineHeight;
    DrawText("7: FULL_THROTTLE   (Throttle 1.0)", x + 20, textY, 16, BLACK); textY += lineHeight;
    DrawText("8: NO_CHANGE       (Keep current)", x + 20, textY, 16, BLACK); textY += lineHeight;
}

std::string DQNVisualizer::getActionName(DQNEnvironment::Action action) {
    switch(action) {
        case DQNEnvironment::Action::STRONG_BRAKE: return "STRONG_BRAKE";
        case DQNEnvironment::Action::MEDIUM_BRAKE: return "MEDIUM_BRAKE";
        case DQNEnvironment::Action::LIGHT_BRAKE: return "LIGHT_BRAKE";
        case DQNEnvironment::Action::COAST: return "COAST";
        case DQNEnvironment::Action::LIGHT_THROTTLE: return "LIGHT_THROTTLE";
        case DQNEnvironment::Action::MEDIUM_THROTTLE: return "MEDIUM_THROTTLE";
        case DQNEnvironment::Action::STRONG_THROTTLE: return "STRONG_THROTTLE";
        case DQNEnvironment::Action::FULL_THROTTLE: return "FULL_THROTTLE";
        case DQNEnvironment::Action::NO_CHANGE: return "NO_CHANGE";
        default: return "UNKNOWN";
    }
}

} // namespace CarGame