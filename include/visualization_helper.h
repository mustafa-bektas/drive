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
 * Focused on current metrics, not training history
 */
class VisualizationHelper {
public:
    VisualizationHelper(int historySize = 120);
    
    // Update speed history (for visualization)
    void updateSpeedHistory(const Car& car, float targetSpeed);
    
    // Draw the visualization UI
    void drawUI(const Car& car, float currentSpeed, float targetSpeed, 
               float simulationSpeed, bool paused, bool modelLoaded);
    
    // Reset metrics (call when resetting simulation)
    void reset();
    
    // Toggle UI visibility
    void toggleUI();
    
private:
    // Speed history for graph
    std::deque<float> speedHistory;
    std::deque<float> targetSpeedHistory;
    
    // Config
    int historySize;
    bool showUI;
    
    // Helper drawing methods
    void drawCompactInfoPanel(int x, int y, int width, int height, 
                             const Car& car, float currentSpeed, 
                             float targetSpeed, float simulationSpeed, bool paused, 
                             bool modelLoaded);
    void drawCompactSpeedGraph(int x, int y, int width, int height, float targetSpeed);
    void drawMinimalHelp(int x, int y, const char* helpText);
    
    // Helper to get action name 
    std::string getActionName(DQNEnvironment::Action action);
    
    // Helper to get color based on value relative to target 
    Color getSpeedColor(float speed, float target);
};

} // namespace CarGame