#pragma once

#include "raylib.h"
#include "car.h"
#include "dqn_environment.h"
#include <vector>
#include <deque>
#include <string>

namespace CarGame {

/**
 * Helper class for visualizing the car simulation 
 * Includes metrics for both speed control and lane following
 */
class VisualizationHelper {
public:
    VisualizationHelper(int historySize = 120);
    
    // Update history for visualization
    void updateSpeedHistory(const Car& car, float targetSpeed);
    void updateLaneHistory(const Car& car, float lateralDeviation, float laneWidth);
    
    // Draw the visualization UI
    void drawUI(const Car& car, float currentSpeed, float targetSpeed, 
               float lateralDeviation, float laneWidth,
               float simulationSpeed, bool paused, bool modelLoaded);
    
    // Reset metrics (call when resetting simulation)
    void reset();
    
    // Toggle UI visibility
    void toggleUI();
    
private:
    // Speed history for graph
    std::deque<float> speedHistory;
    std::deque<float> targetSpeedHistory;
    
    // Lane following metrics
    std::deque<float> lateralDeviationHistory;
    std::deque<float> headingErrorHistory;
    
    // Config
    int historySize;
    bool showUI;
    
    // Helper drawing methods
    void drawCompactInfoPanel(int x, int y, int width, int height, 
                             const Car& car, float currentSpeed, 
                             float targetSpeed, float lateralDeviation,
                             float laneWidth, float simulationSpeed, bool paused, 
                             bool modelLoaded);
    void drawCompactSpeedGraph(int x, int y, int width, int height, float targetSpeed);
    void drawLaneDeviationGraph(int x, int y, int width, int height, float laneWidth);
    void drawMinimalHelp(int x, int y, const char* helpText);
    
    // Helper to get action name 
    std::string getActionName(DQNEnvironment::Action action);
    
    // Helper to get color based on value relative to target 
    Color getSpeedColor(float speed, float target);
    Color getLaneDeviationColor(float deviation, float laneWidth);
};

} // namespace CarGame