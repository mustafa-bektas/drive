#include "../include/visualization_helper.h"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace CarGame {

VisualizationHelper::VisualizationHelper(int historySize)
    : historySize(historySize), showUI(true) {
}

void VisualizationHelper::updateSpeedHistory(const Car& car, float targetSpeed) {
    // Add values to history
    speedHistory.push_back(car.speed);
    targetSpeedHistory.push_back(targetSpeed);
    
    // Limit history size
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
        // Only show a minimal help indicator when UI is hidden
        DrawText("Press TAB to show UI", 10, 10, 20, Fade(DARKGRAY, 0.7f));
        return;
    }
    
    int screenWidth = GetScreenWidth();
    int screenHeight = GetScreenHeight();
    
    // Draw a compact info panel in the top-left corner
    drawCompactInfoPanel(10, 10, 300, 65, car, currentSpeed, targetSpeed, 
                       simulationSpeed, paused, modelLoaded);
    
    // Draw small speed graph in the bottom-left corner
    drawCompactSpeedGraph(10, screenHeight - 110, 280, 100, targetSpeed);
    
    // Draw help at the bottom
    drawMinimalHelp(10, screenHeight - 25, "SPACE: Pause | R: Reset | UP/DOWN: Speed | TAB: Toggle UI | F: Fullscreen | ESC: Exit");
}

void VisualizationHelper::drawCompactInfoPanel(int x, int y, int width, int height, 
                                             const Car& car, float currentSpeed, 
                                             float targetSpeed, 
                                             float simulationSpeed, bool paused, bool modelLoaded) {
    // Semi-transparent background
    DrawRectangle(x, y, width, height, Fade(LIGHTGRAY, 0.7f));
    
    // Title
    Color titleColor = modelLoaded ? DARKGREEN : MAROON;
    DrawText("DQN Speed Control", x + 10, y + 5, 16, titleColor);
    
    // Main metrics - three columns
    int col1X = x + 10;
    int col2X = x + 110;
    int col3X = x + 210;
    int metricsY = y + 30;
    
    // Column 1
    DrawText(TextFormat("Speed: %.1f", car.speed * 3.6f), col1X, metricsY, 15, 
            getSpeedColor(car.speed, targetSpeed));
    
    // Column 2
    DrawText(TextFormat("Target: %.1f", targetSpeed * 3.6f), col2X, metricsY, 15, DARKGREEN);
    
    // Column 3
    DrawText(TextFormat("RPM: %.0f", car.engineSpeed), col3X, metricsY, 15, DARKBLUE);
    
    // Status line
    Color statusColor = paused ? ORANGE : DARKGREEN;
    DrawText(TextFormat("Throttle: %.2f | Brake: %.2f | %s", 
                      car.throttle, car.brake, paused ? "PAUSED" : "RUNNING"), 
            x + 10, metricsY + 22, 15, statusColor);
}

void VisualizationHelper::drawCompactSpeedGraph(int x, int y, int width, int height, float targetSpeed) {
    // Semi-transparent background
    DrawRectangle(x, y, width, height, Fade(LIGHTGRAY, 0.7f));
    
    // Title
    DrawText("Speed History", x + 10, y + 5, 15, DARKBLUE);
    
    // Draw history graph
    if (speedHistory.size() > 1) {
        int graphX = x + 10;
        int graphY = y + 25;
        int graphWidth = width - 20;
        int graphHeight = height - 35;
        
        // Background
        DrawRectangle(graphX, graphY, graphWidth, graphHeight, Fade(WHITE, 0.9f));
        
        // Find data bounds
        float maxSpeed = targetSpeed * 1.5f; // Show up to 150% of target speed
        
        float xScale = (float)graphWidth / (historySize - 1);
        float yScale = (float)graphHeight / maxSpeed;
        
        // Draw target speed line
        DrawLine(
            graphX, graphY + graphHeight - (targetSpeed * yScale),
            graphX + graphWidth, graphY + graphHeight - (targetSpeed * yScale),
            Fade(DARKGREEN, 0.7f)
        );
        
        // Draw speed history
        for (size_t i = 0; i < speedHistory.size() - 1; i++) {
            float x1 = graphX + i * xScale;
            float y1 = graphY + graphHeight - (speedHistory[i] * yScale);
            float x2 = graphX + (i + 1) * xScale;
            float y2 = graphY + graphHeight - (speedHistory[i + 1] * yScale);
            
            DrawLine(x1, y1, x2, y2, Fade(RED, 0.8f));
        }
        
        // Draw current speed label
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
    // Calculate color based on how close we are to target
    float ratio = speed / target;
    
    if (ratio < 0.8f) {
        // Too slow - yellow to red
        unsigned char r = 255;
        unsigned char g = static_cast<unsigned char>(255 * (ratio / 0.8f));
        return {r, g, 0, 255};
    } else if (ratio <= 1.2f) {
        // Good range - green
        float adjustment = std::abs(ratio - 1.0f) / 0.2f;
        unsigned char g = 180 + static_cast<unsigned char>(75 * (1.0f - adjustment));
        return {0, g, 0, 255};
    } else {
        // Too fast - yellow to red
        float excess = (ratio - 1.2f) / 0.8f;
        if (excess > 1.0f) excess = 1.0f;
        unsigned char r = 255;
        unsigned char g = static_cast<unsigned char>(255 * (1.0f - excess));
        return {r, g, 0, 255};
    }
}

} // namespace CarGame