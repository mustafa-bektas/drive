#include "../include/rendering.h"
#include "../include/enhanced_episode_manager.h"
#include <string>
#include <array>
#include <vector>
#include <functional>
#include <algorithm>
#include "rlgl.h"

#pragma once

namespace CarGame {

// UI constants for enhanced visualization
namespace EnhancedUI {
    constexpr int FontSize = 20;
    constexpr int SmallFontSize = 16;
    constexpr int LineHeight = 24;
    constexpr int PanelMargin = 10;
    constexpr int SectionSpacing = 5;
    constexpr int PanelWidth = 400;
    constexpr int LearningPanelHeight = 400;
    
    const Color TextColor = BLACK;
    const Color HeaderColor = DARKBLUE;
    const Color SectionColor = DARKGRAY;
    const Color PanelColor = { 200, 200, 200, 180 };
    const Color RewardColor = GREEN;
    const Color NegativeRewardColor = RED;
    const Color SpeedErrorColor = ORANGE;
    const Color TargetSpeedColor = BLUE;
    const Color ExplorationColor = PURPLE;
}

// Enhanced drawer methods to add to the Renderer class
void drawEnhancedRLStats(const Car& car, const RLAgent& agent, const EnhancedEpisodeManager& manager) {
    int textX = 20;
    int textY = 20;
    
    // Draw panel background
    DrawRectangle(
        textX - EnhancedUI::PanelMargin, 
        textY - EnhancedUI::PanelMargin, 
        EnhancedUI::PanelWidth, 
        EnhancedUI::LearningPanelHeight, 
        EnhancedUI::PanelColor
    );
    
    // Panel title
    DrawText("REINFORCEMENT LEARNING STATUS", textX, textY, EnhancedUI::FontSize, EnhancedUI::HeaderColor);
    textY += EnhancedUI::LineHeight + EnhancedUI::SectionSpacing;
    
    // RL stats
    DrawText(TextFormat("Episode: %d", manager.getEpisodeCount()), 
             textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight;
    
    DrawText(TextFormat("Current Reward: %.2f", manager.getCurrentReward()), 
             textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight;
    
    DrawText(TextFormat("Episode Reward: %.2f", manager.getCumulativeReward()), 
             textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight;
    
    DrawText(TextFormat("Avg Reward: %.2f", manager.getAverageReward()), 
             textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight;
    
    DrawText(TextFormat("Best Reward: %.2f", manager.getBestReward()), 
             textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight;
    
    DrawText(TextFormat("Success Rate: %.1f%%", manager.getSuccessRate() * 100.0f), 
             textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight;
    
    DrawText(TextFormat("Exploration Rate: %.2f", agent.getExplorationRate()), 
             textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight;
    
    DrawText(TextFormat("Q-Table Size: %d", agent.getQTableSize()), 
             textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight;
    
    DrawText(TextFormat("Target Speed: %.1f km/h", manager.getTargetSpeed() * 3.6f), 
             textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight;
    
    DrawText(TextFormat("Current Speed: %.1f km/h", car.speed * 3.6f), 
             textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight;
    
    DrawText(TextFormat("Speed Error: %.1f km/h", manager.getSpeedError() * 3.6f), 
             textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight * 2;
    
    // Draw time left in episode
    float timeLeft = manager.isEpisodeComplete() ? 0.0f : 
                    manager.getEpisodeTimer() - manager.getEpisodeTimer();
    
    DrawText("Episode Progress:", textX, textY, EnhancedUI::FontSize, EnhancedUI::TextColor);
    textY += EnhancedUI::LineHeight;
    
    // Draw progress bar
    float progress = manager.getEpisodeTimer() / 10.0f; // Assuming 10-second episodes
    progress = std::min(1.0f, std::max(0.0f, progress));
    
    int barWidth = EnhancedUI::PanelWidth - 40;
    int barHeight = 20;
    
    DrawRectangle(textX, textY, barWidth, barHeight, LIGHTGRAY);
    DrawRectangle(textX, textY, (int)(barWidth * progress), barHeight, BLUE);
    DrawRectangleLines(textX, textY, barWidth, barHeight, DARKGRAY);
    
    textY += barHeight + EnhancedUI::LineHeight;
}

void drawLearningGraphs(const EnhancedEpisodeManager& manager, const RLAgent& agent) {
    const int graphX = 440;
    const int graphY = 20;
    const int graphWidth = 500;
    const int graphHeight = 200;
    const int graphMargin = 10;
    
    // Draw panel background
    DrawRectangle(
        graphX - EnhancedUI::PanelMargin, 
        graphY - EnhancedUI::PanelMargin, 
        graphWidth + EnhancedUI::PanelMargin * 2, 
        graphHeight * 2 + EnhancedUI::PanelMargin * 3 + EnhancedUI::LineHeight * 2, 
        EnhancedUI::PanelColor
    );
    
    // Draw reward history graph
    DrawText("Reward History", graphX, graphY, EnhancedUI::FontSize, EnhancedUI::HeaderColor);
    
    // Draw graph background
    DrawRectangleLines(graphX, graphY + EnhancedUI::LineHeight, graphWidth, graphHeight, DARKGRAY);
    DrawRectangle(graphX, graphY + EnhancedUI::LineHeight, graphWidth, graphHeight, {240, 240, 240, 255});
    
    // Draw horizontal reference lines
    for (int i = 1; i < 4; i++) {
        float y = graphY + EnhancedUI::LineHeight + graphHeight * i / 4.0f;
        DrawLine(graphX, y, graphX + graphWidth, y, {200, 200, 200, 128});
    }
    
    // Draw reward graph
    const auto& rewardHistory = manager.getRewardHistory();
    if (!rewardHistory.empty()) {
        // Find max and min values for scaling
        float maxReward = *std::max_element(rewardHistory.begin(), rewardHistory.end());
        float minReward = *std::min_element(rewardHistory.begin(), rewardHistory.end());
        
        // Ensure range is at least 1.0 for visibility
        float range = std::max(1.0f, maxReward - minReward);
        
        // Scale and draw the lines
        const int maxPoints = std::min((int)rewardHistory.size(), graphWidth);
        const int startIdx = rewardHistory.size() <= maxPoints ? 0 : rewardHistory.size() - maxPoints;
        
        for (int i = 0; i < maxPoints - 1; i++) {
            float reward1 = rewardHistory[startIdx + i];
            float reward2 = rewardHistory[startIdx + i + 1];
            
            float x1 = graphX + (float)i / maxPoints * graphWidth;
            float x2 = graphX + (float)(i + 1) / maxPoints * graphWidth;
            
            float y1 = graphY + EnhancedUI::LineHeight + graphHeight - 
                       (reward1 - minReward) / range * graphHeight * 0.8f - 10;
            float y2 = graphY + EnhancedUI::LineHeight + graphHeight - 
                       (reward2 - minReward) / range * graphHeight * 0.8f - 10;
            
            Color lineColor = reward1 >= 0 ? EnhancedUI::RewardColor : EnhancedUI::NegativeRewardColor;
            DrawLine(x1, y1, x2, y2, lineColor);
        }
        
        // Draw scale
        DrawText(TextFormat("Max: %.1f", maxReward), 
                 graphX + graphWidth - 100, graphY + EnhancedUI::LineHeight + 10, 
                 EnhancedUI::SmallFontSize, EnhancedUI::TextColor);
        DrawText(TextFormat("Min: %.1f", minReward), 
                 graphX + graphWidth - 100, graphY + EnhancedUI::LineHeight + graphHeight - 20, 
                 EnhancedUI::SmallFontSize, EnhancedUI::TextColor);
    }
    
    // Draw speed error graph
    int speedGraphY = graphY + graphHeight + EnhancedUI::LineHeight + EnhancedUI::PanelMargin;
    DrawText("Speed Control Performance", graphX, speedGraphY, EnhancedUI::FontSize, EnhancedUI::HeaderColor);
    
    speedGraphY += EnhancedUI::LineHeight;
    
    // Draw graph background
    DrawRectangleLines(graphX, speedGraphY, graphWidth, graphHeight, DARKGRAY);
    DrawRectangle(graphX, speedGraphY, graphWidth, graphHeight, {240, 240, 240, 255});
    
    // Draw center line (zero error)
    DrawLine(graphX, speedGraphY + graphHeight/2, graphX + graphWidth, speedGraphY + graphHeight/2, 
             {100, 100, 100, 200});
    
    // Draw speed error graph
    const auto& speedErrorHistory = manager.getSpeedErrorHistory();
    if (!speedErrorHistory.empty()) {
        // Set fixed range for speed error
        float maxError = 10.0f; // 10 m/s error range
        
        // Scale and draw the lines
        const int maxPoints = std::min((int)speedErrorHistory.size(), graphWidth);
        const int startIdx = speedErrorHistory.size() <= maxPoints ? 0 : speedErrorHistory.size() - maxPoints;
        
        for (int i = 0; i < maxPoints - 1; i++) {
            float error1 = speedErrorHistory[startIdx + i];
            float error2 = speedErrorHistory[startIdx + i + 1];
            
            // Limit to range for display
            error1 = std::max(-maxError, std::min(maxError, error1));
            error2 = std::max(-maxError, std::min(maxError, error2));
            
            float x1 = graphX + (float)i / maxPoints * graphWidth;
            float x2 = graphX + (float)(i + 1) / maxPoints * graphWidth;
            
            // Center at middle of graph (zero error)
            float y1 = speedGraphY + graphHeight/2 - (error1 / maxError) * (graphHeight/2 - 10);
            float y2 = speedGraphY + graphHeight/2 - (error2 / maxError) * (graphHeight/2 - 10);
            
            DrawLine(x1, y1, x2, y2, EnhancedUI::SpeedErrorColor);
        }
        
        // Draw target speed reference
        DrawLine(graphX, speedGraphY + graphHeight/2, graphX + graphWidth, speedGraphY + graphHeight/2, 
                 EnhancedUI::TargetSpeedColor);
                 
        // Draw scale
        DrawText(TextFormat("+%.1f m/s", maxError), 
                 graphX + 10, speedGraphY + 10, 
                 EnhancedUI::SmallFontSize, EnhancedUI::TextColor);
        DrawText("0 m/s", 
                 graphX + 10, speedGraphY + graphHeight/2 - 10, 
                 EnhancedUI::SmallFontSize, EnhancedUI::TextColor);
        DrawText(TextFormat("-%.1f m/s", maxError), 
                 graphX + 10, speedGraphY + graphHeight - 20, 
                 EnhancedUI::SmallFontSize, EnhancedUI::TextColor);
    }
    
    // Draw exploration rate as a line on the speed graph
    const auto& explorationHistory = manager.getExplorationRateHistory();
    if (!explorationHistory.empty()) {
        const int maxPoints = std::min((int)explorationHistory.size(), graphWidth);
        const int startIdx = explorationHistory.size() <= maxPoints ? 0 : explorationHistory.size() - maxPoints;
        
        for (int i = 0; i < maxPoints - 1; i++) {
            float rate1 = explorationHistory[startIdx + i];
            float rate2 = explorationHistory[startIdx + i + 1];
            
            float x1 = graphX + (float)i / maxPoints * graphWidth;
            float x2 = graphX + (float)(i + 1) / maxPoints * graphWidth;
            
            float y1 = speedGraphY + graphHeight - rate1 * graphHeight;
            float y2 = speedGraphY + graphHeight - rate2 * graphHeight;
            
            DrawLine(x1, y1, x2, y2, EnhancedUI::ExplorationColor);
        }
        
        // Add legend
        DrawText("Exploration Rate", graphX + graphWidth - 150, speedGraphY + graphHeight - 20, 
                 EnhancedUI::SmallFontSize, EnhancedUI::ExplorationColor);
    }
}

// Draw controls and actions
void drawActionSpace(const RLAgent& agent, const Car& car) {
    const int startX = 960;
    const int startY = 50;
    const int cellSize = 60;
    const int gridWidth = 3;
    const int gridHeight = 3;
    
    // Draw panel background
    DrawRectangle(
        startX - EnhancedUI::PanelMargin, 
        startY - EnhancedUI::PanelMargin, 
        gridWidth * cellSize + EnhancedUI::PanelMargin * 2, 
        gridHeight * cellSize + EnhancedUI::PanelMargin * 2 + EnhancedUI::LineHeight, 
        EnhancedUI::PanelColor
    );
    
    // Draw header
    DrawText("Action Space", startX, startY - 30, EnhancedUI::FontSize, EnhancedUI::HeaderColor);
    
    // Draw throttle label (vertical axis)
    DrawText("Throttle", startX - 50, startY + cellSize * gridHeight / 2 - 10, 
             EnhancedUI::SmallFontSize, EnhancedUI::TextColor);
    
    // Draw brake label (horizontal axis)
    DrawText("Brake", startX + cellSize * gridWidth / 2 - 20, startY + cellSize * gridHeight + 10, 
             EnhancedUI::SmallFontSize, EnhancedUI::TextColor);
    
    // Get current action for highlighting
    bool actionHighlighted = false;
    int currentActionIdx = -1;
    
    if (car.throttle >= 0.0f && car.brake >= 0.0f) {
        // Find closest action in action space
        const auto& actionSpace = agent.getActionSpace();
        float minDist = std::numeric_limits<float>::max();
        
        for (size_t i = 0; i < actionSpace.size(); i++) {
            float throttleDiff = car.throttle - actionSpace[i][0];
            float brakeDiff = car.brake - actionSpace[i][1];
            float dist = throttleDiff * throttleDiff + brakeDiff * brakeDiff;
            
            if (dist < minDist) {
                minDist = dist;
                currentActionIdx = i;
            }
        }
        
        if (currentActionIdx >= 0 && minDist < 0.1f) {
            actionHighlighted = true;
        }
    }
    
    // Draw action grid
    for (int y = 0; y < gridHeight; y++) {
        for (int x = 0; x < gridWidth; x++) {
            int idx = y * gridWidth + x;
            if (idx >= agent.getActionSpaceSize()) continue;
            
            int cellX = startX + x * cellSize;
            int cellY = startY + (gridHeight - 1 - y) * cellSize; // Invert Y axis
            
            // Draw cell background
            Color cellColor = LIGHTGRAY;
            if (actionHighlighted && idx == currentActionIdx) {
                cellColor = GREEN;
            }
            
            DrawRectangle(cellX, cellY, cellSize, cellSize, cellColor);
            DrawRectangleLines(cellX, cellY, cellSize, cellSize, DARKGRAY);
            
            // Draw action values (if available)
            const auto& actionSpace = agent.getActionSpace();
            if (idx < actionSpace.size()) {
                float throttle = actionSpace[idx][0];
                float brake = actionSpace[idx][1];
                
                DrawText(TextFormat("T:%.1f", throttle), cellX + 5, cellY + 10, 
                         EnhancedUI::SmallFontSize - 2, DARKGRAY);
                DrawText(TextFormat("B:%.1f", brake), cellX + 5, cellY + 30, 
                         EnhancedUI::SmallFontSize - 2, DARKGRAY);
            }
        }
    }
}

} // namespace CarGame