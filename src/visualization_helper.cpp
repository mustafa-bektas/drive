#include "../include/visualization_helper.h"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace CarGame {

VisualizationHelper::VisualizationHelper(int historySize)
    : historySize(historySize), showUI(true) {
}

void VisualizationHelper::updateSpeedHistory(const Car& car, float targetSpeed) {
    // add to history
    speedHistory.push_back(car.speed);
    targetSpeedHistory.push_back(targetSpeed);
    
    // limit history size
    if (speedHistory.size() > historySize) {
        speedHistory.pop_front();
    }
    if (targetSpeedHistory.size() > historySize) {
        targetSpeedHistory.pop_front();
    }
}

void VisualizationHelper::reset() {
    speedHistory.clear();
    targetSpeedHistory.clear();
}

void VisualizationHelper::toggleUI() {
    showUI = !showUI;
}

void VisualizationHelper::drawUI(const Car& car, float currentSpeed, float targetSpeed, 
                               float simulationSpeed, bool paused, bool modelLoaded) {
    if (!showUI) {
        // minimal help when ui hidden
        DrawText("Press TAB to show UI", 10, 10, 20, Fade(DARKGRAY, 0.7f));
        return;
    }
    
    int screenWidth = GetScreenWidth();
    int screenHeight = GetScreenHeight();
    
    // top-left info panel
    drawCompactInfoPanel(10, 10, 300, 65, car, currentSpeed, targetSpeed, 
                       simulationSpeed, paused, modelLoaded);
    
    // bottom-left speed graph
    drawCompactSpeedGraph(10, screenHeight - 110, 280, 100, targetSpeed);
    
    // help at bottom
    drawMinimalHelp(10, screenHeight - 25, "SPACE: Pause | R: Reset | PgUp/PgDn: Speed | L: Lane Keeping | TAB: Toggle UI | F: Fullscreen | ESC: Exit");
}

void VisualizationHelper::drawLaneInfo(const Car& car, float laneWidth, bool laneKeepingActive) {
    if (!showUI) {
        return;
    }
    
    int screenWidth = GetScreenWidth();
    int screenHeight = GetScreenHeight();
    
    // top-right lane info
    int panelWidth = 280;
    int panelHeight = 120;
    int x = screenWidth - panelWidth - 10;
    int y = 10;
    
    // bg panel
    DrawRectangle(x, y, panelWidth, panelHeight, Fade(LIGHTGRAY, 0.7f));
    
    // title
    Color titleColor = laneKeepingActive ? DARKGREEN : DARKBLUE;
    DrawText("Lane Position", x + 10, y + 5, 16, titleColor);
    
    // lane metrics
    float lateralPosition = car.position.x;
    float laneCenter = 0.0f;
    float laneEdgeLeft = laneCenter - laneWidth/2;
    float laneEdgeRight = laneCenter + laneWidth/2;
    
    // text metrics
    DrawText(TextFormat("Position: %.2f m", lateralPosition), x + 10, y + 30, 15, BLACK);
    DrawText(TextFormat("Lane Width: %.1f m", laneWidth), x + 10, y + 50, 15, BLACK);
    DrawText(TextFormat("Distance to Center: %.2f m", std::abs(lateralPosition - laneCenter)), 
             x + 10, y + 70, 15, BLACK);
    
    // lane status
    const char* statusText;
    Color statusColor;
    
    if (std::abs(lateralPosition) < 0.5f) {
        statusText = "Status: CENTERED";
        statusColor = DARKGREEN;
    } else if (std::abs(lateralPosition) < laneWidth/2) {
        statusText = "Status: IN LANE";
        statusColor = BLUE;
    } else {
        statusText = "Status: OUT OF LANE";
        statusColor = RED;
    }
    
    DrawText(statusText, x + 10, y + 90, 15, statusColor);
    
    // visual lane indicator
    int indicatorY = y + panelHeight + 10;
    int indicatorHeight = 30;
    int indicatorWidth = panelWidth;
    
    // lane bg
    DrawRectangle(x, indicatorY, indicatorWidth, indicatorHeight, LIGHTGRAY);
    
    // lane markings
    int laneLeft = x + indicatorWidth/2 - indicatorWidth/3;
    int laneRight = x + indicatorWidth/2 + indicatorWidth/3;
    
    // center and edges
    DrawLine(x + indicatorWidth/2, indicatorY, x + indicatorWidth/2, indicatorY + indicatorHeight, 
             Fade(DARKGRAY, 0.5f));
    DrawRectangle(laneLeft, indicatorY, 5, indicatorHeight, WHITE);
    DrawRectangle(laneRight, indicatorY, 5, indicatorHeight, WHITE);
    
    // car position indicator
    float normalizedPos = (lateralPosition - laneEdgeLeft) / laneWidth;
    int carPosX = x + static_cast<int>(normalizedPos * indicatorWidth);
    
    DrawTriangle(
        {static_cast<float>(carPosX), static_cast<float>(indicatorY)},
        {static_cast<float>(carPosX - 10), static_cast<float>(indicatorY + indicatorHeight)},
        {static_cast<float>(carPosX + 10), static_cast<float>(indicatorY + indicatorHeight)},
        RED
    );
}

void VisualizationHelper::drawCompactInfoPanel(int x, int y, int width, int height, 
                                             const Car& car, float currentSpeed, 
                                             float targetSpeed, 
                                             float simulationSpeed, bool paused, bool modelLoaded) {
    // semi-transparent bg
    DrawRectangle(x, y, width, height, Fade(LIGHTGRAY, 0.7f));
    
    // title
    Color titleColor = modelLoaded ? DARKGREEN : MAROON;
    DrawText("DQN Speed Control", x + 10, y + 5, 16, titleColor);
    
    // main metrics - three cols
    int col1X = x + 10;
    int col2X = x + 110;
    int col3X = x + 210;
    int metricsY = y + 30;
    
    // col 1
    DrawText(TextFormat("Speed: %.1f", car.speed * 3.6f), col1X, metricsY, 15, 
            getSpeedColor(car.speed, targetSpeed));
    
    // col 2
    DrawText(TextFormat("Target: %.1f", targetSpeed * 3.6f), col2X, metricsY, 15, DARKGREEN);
    
    // col 3
    DrawText(TextFormat("RPM: %.0f", car.engineSpeed), col3X, metricsY, 15, DARKBLUE);
    
    // status line
    Color statusColor = paused ? ORANGE : DARKGREEN;
    DrawText(TextFormat("Throttle: %.2f | Brake: %.2f | %s", 
                      car.throttle, car.brake, paused ? "PAUSED" : "RUNNING"), 
            x + 10, metricsY + 22, 15, statusColor);
}

void VisualizationHelper::drawCompactSpeedGraph(int x, int y, int width, int height, float targetSpeed) {
    // semi-transparent bg
    DrawRectangle(x, y, width, height, Fade(LIGHTGRAY, 0.7f));
    
    // title
    DrawText("Speed History", x + 10, y + 5, 15, DARKBLUE);
    
    // graph
    if (speedHistory.size() > 1) {
        int graphX = x + 10;
        int graphY = y + 25;
        int graphWidth = width - 20;
        int graphHeight = height - 35;
        
        // bg
        DrawRectangle(graphX, graphY, graphWidth, graphHeight, Fade(WHITE, 0.9f));
        
        // data bounds
        float maxSpeed = targetSpeed * 1.5f; // 150% of target
        
        float xScale = (float)graphWidth / (historySize - 1);
        float yScale = (float)graphHeight / maxSpeed;
        
        // target line
        DrawLine(
            graphX, graphY + graphHeight - (targetSpeed * yScale),
            graphX + graphWidth, graphY + graphHeight - (targetSpeed * yScale),
            Fade(DARKGREEN, 0.7f)
        );
        
        // speed history
        for (size_t i = 0; i < speedHistory.size() - 1; i++) {
            float x1 = graphX + i * xScale;
            float y1 = graphY + graphHeight - (speedHistory[i] * yScale);
            float x2 = graphX + (i + 1) * xScale;
            float y2 = graphY + graphHeight - (speedHistory[i + 1] * yScale);
            
            DrawLine(x1, y1, x2, y2, Fade(RED, 0.8f));
        }
        
        // current speed label
        if (!speedHistory.empty()) {
            DrawText(TextFormat("%.1f km/h", speedHistory.back() * 3.6f), 
                    graphX + graphWidth - 70, graphY + 5, 15, RED);
        }
    }
}

void VisualizationHelper::drawMinimalHelp(int x, int y, const char* helpText) {
    DrawText(helpText, x, y, 15, Fade(DARKGRAY, 0.7f));
}

std::string VisualizationHelper::getActionName(DQNEnvironment::Action action) {
    switch(action) {
        case DQNEnvironment::STRONG_BRAKE: return "STRONG BRAKE";
        case DQNEnvironment::MEDIUM_BRAKE: return "MEDIUM BRAKE";
        case DQNEnvironment::LIGHT_BRAKE: return "LIGHT BRAKE";
        case DQNEnvironment::COAST: return "COAST";
        case DQNEnvironment::LIGHT_THROTTLE: return "LIGHT THROTTLE";
        case DQNEnvironment::MEDIUM_THROTTLE: return "MEDIUM THROTTLE";
        case DQNEnvironment::STRONG_THROTTLE: return "STRONG THROTTLE";
        case DQNEnvironment::FULL_THROTTLE: return "FULL THROTTLE";
        case DQNEnvironment::NO_CHANGE: return "NO CHANGE";
        default: return "UNKNOWN";
    }
}

Color VisualizationHelper::getSpeedColor(float speed, float target) {
    // color based on closeness to target
    float ratio = speed / target;
    
    if (ratio < 0.8f) {
        // too slow - yellow to red
        unsigned char r = 255;
        unsigned char g = static_cast<unsigned char>(255 * (ratio / 0.8f));
        return {r, g, 0, 255};
    } else if (ratio <= 1.2f) {
        // good range - green
        float adjustment = std::abs(ratio - 1.0f) / 0.2f;
        unsigned char g = 180 + static_cast<unsigned char>(75 * (1.0f - adjustment));
        return {0, g, 0, 255};
    } else {
        // too fast - yellow to red
        float excess = (ratio - 1.2f) / 0.8f;
        if (excess > 1.0f) excess = 1.0f;
        unsigned char r = 255;
        unsigned char g = static_cast<unsigned char>(255 * (1.0f - excess));
        return {r, g, 0, 255};
    }
}

} // namespace CarGame