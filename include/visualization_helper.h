#pragma once

#include "raylib.h"
#include "car.h"
#include "dqn_environment.h"
#include <vector>
#include <deque>
#include <string>

namespace CarGame {

/**
 * Helper class for visualizing the car simulation with smooth metrics
 */
class VisualizationHelper {
public:
    VisualizationHelper(int historySize = 120);
    
    // Update metrics (should be called each simulation step)
    void updateMetrics(const Car& car, float reward, DQNEnvironment::Action action, float targetSpeed);
    
    // Draw the visualization UI
    void drawUI(const Car& car, float totalReward, int stepCount, float targetSpeed, 
               float simulationSpeed, bool paused, bool modelLoaded);
    
    // Reset metrics (call when resetting simulation)
    void reset();
    
    // Toggle UI visibility
    void toggleUI();
    
private:
    // History tracking
    std::deque<float> speedHistory;
    std::deque<float> rewardHistory;
    std::deque<float> targetSpeedHistory;
    std::deque<DQNEnvironment::Action> actionHistory;
    
    // Smoothed values
    float smoothedSpeed;
    float smoothedReward;
    
    // Config
    int historySize;
    bool showUI;
    
    // Helper drawing methods
    void drawCompactInfoPanel(int x, int y, int width, int height, 
                             const Car& car, float totalReward, int stepCount, 
                             float targetSpeed, float simulationSpeed, bool paused, 
                             bool modelLoaded);
    void drawCompactSpeedGraph(int x, int y, int width, int height, float targetSpeed);
    void drawMinimalHelp(int x, int y, const char* helpText);
    
    // Helper to get action name 
    std::string getActionName(DQNEnvironment::Action action);
    
    // Helper to get color based on value relative to target 
    Color getSpeedColor(float speed, float target);
    
    // Helper to draw a horizontal bar indicator
    void drawBarIndicator(int x, int y, int width, int height, float value, 
                          float minValue, float maxValue, Color color);
};

} // namespace CarGame