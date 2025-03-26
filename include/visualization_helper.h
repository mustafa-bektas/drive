#pragma once

#include "raylib.h"
#include "car.h"
#include "dqn_environment.h"
#include <vector>
#include <deque>
#include <string>

namespace CarGame {

// viz helper - shows current metrics only
class VisualizationHelper {
public:
    VisualizationHelper(int historySize = 120);
    
    // update speed history
    void updateSpeedHistory(const Car& car, float targetSpeed);
    
    // draw ui
    void drawUI(const Car& car, float currentSpeed, float targetSpeed, 
               float simulationSpeed, bool paused, bool modelLoaded);
    
    // draw lane info
    void drawLaneInfo(const Car& car, float laneWidth, bool laneKeepingActive);
    
    // reset metrics
    void reset();
    
    // toggle ui
    void toggleUI();
    
private:
    // history for graph
    std::deque<float> speedHistory;
    std::deque<float> targetSpeedHistory;
    
    // config
    int historySize;
    bool showUI;
    
    // helper draw methods
    void drawCompactInfoPanel(int x, int y, int width, int height, 
                             const Car& car, float currentSpeed, 
                             float targetSpeed, float simulationSpeed, bool paused, 
                             bool modelLoaded);
    void drawCompactSpeedGraph(int x, int y, int width, int height, float targetSpeed);
    void drawMinimalHelp(int x, int y, const char* helpText);
    
    // get action name
    std::string getActionName(DQNEnvironment::Action action);
    
    // get color based on value vs target
    Color getSpeedColor(float speed, float target);
};

} // namespace CarGame