#include "../include/visualization_helper.h"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace CarGame {

VisualizationHelper::VisualizationHelper(int historySize)
    : historySize(historySize), showUI(true) {
}

void VisualizationHelper::updateSpeedHistory(const Car& car, float targetSpeed) {
    // Add values to speed history
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

void VisualizationHelper::updateLaneHistory(const Car& car, float lateralDeviation, float laneWidth) {
    // Add values to lane history
    lateralDeviationHistory.push_back(lateralDeviation);
    headingErrorHistory.push_back(car.rotation);  // Using rotation as a simple proxy for heading error
    
    // Limit history size
    if (lateralDeviationHistory.size() > historySize) {
        lateralDeviationHistory.pop_front();
    }
    if (headingErrorHistory.size() > historySize) {
        headingErrorHistory.pop_front();
    }
}

void VisualizationHelper::reset() {
    speedHistory.clear();
    targetSpeedHistory.clear();
    lateralDeviationHistory.clear();
    headingErrorHistory.clear();
}

void VisualizationHelper::toggleUI() {
    showUI = !showUI;
}

void VisualizationHelper::drawUI(const Car& car, float currentSpeed, float targetSpeed, 
                               float lateralDeviation, float laneWidth,
                               float simulationSpeed, bool paused, bool modelLoaded) {
    if (!showUI) {
        // Only show a minimal help indicator when UI is hidden
        DrawText("Press TAB to show UI", 10, 10, 20, Fade(DARKGRAY, 0.7f));
        return;
    }
    
    int screenWidth = GetScreenWidth();
    int screenHeight = GetScreenHeight();
    
    // Draw a compact info panel in the top-left corner
    drawCompactInfoPanel(10, 10, 350, 120, car, currentSpeed, targetSpeed, 
                       lateralDeviation, laneWidth, simulationSpeed, paused, modelLoaded);
    
    // Draw small speed graph in the bottom-left corner
    drawCompactSpeedGraph(10, screenHeight - 220, 280, 100, targetSpeed);
    
    // Draw lane deviation graph below speed graph
    drawLaneDeviationGraph(10, screenHeight - 110, 280, 100, laneWidth);
    
    // Draw help at the bottom
    drawMinimalHelp(10, screenHeight - 25, "SPACE: Pause | R: Reset | UP/DOWN: Speed | TAB: Toggle UI | F: Fullscreen | ESC: Exit");
}

void VisualizationHelper::drawCompactInfoPanel(int x, int y, int width, int height, 
                                             const Car& car, float currentSpeed, 
                                             float targetSpeed, float lateralDeviation,
                                             float laneWidth, float simulationSpeed, 
                                             bool paused, bool modelLoaded) {
    // Semi-transparent background
    DrawRectangle(x, y, width, height, Fade(LIGHTGRAY, 0.7f));
    
    // Title
    Color titleColor = modelLoaded ? DARKGREEN : MAROON;
    DrawText("DQN Speed & Lane Control", x + 10, y + 5, 16, titleColor);
    
    // Main metrics - three columns
    int col1X = x + 10;
    int col2X = x + 130;
    int col3X = x + 250;
    int metricsY = y + 30;
    
    // Row 1 - Speed metrics
    DrawText(TextFormat("Speed: %.1f km/h", car.speed * 3.6f), col1X, metricsY, 15, 
            getSpeedColor(car.speed, targetSpeed));
    
    DrawText(TextFormat("Target: %.1f km/h", targetSpeed * 3.6f), col2X, metricsY, 15, DARKGREEN);
    
    DrawText(TextFormat("RPM: %.0f", car.engineSpeed), col3X, metricsY, 15, DARKBLUE);
    
    // Row 2 - Lane metrics
    DrawText(TextFormat("Lane Dev: %.2f m", lateralDeviation), col1X, metricsY + 20, 15, 
            getLaneDeviationColor(lateralDeviation, laneWidth));
    
    DrawText(TextFormat("Lane Width: %.1f m", laneWidth), col2X, metricsY + 20, 15, DARKBLUE);
    
    DrawText(TextFormat("Heading: %.1f°", car.rotation * RAD2DEG), col3X, metricsY + 20, 15, DARKBLUE);
    
    // Row 3 - Controls
    DrawText(TextFormat("Throttle: %.2f", car.throttle), col1X, metricsY + 40, 15, DARKBLUE);
    
    DrawText(TextFormat("Brake: %.2f", car.brake), col2X, metricsY + 40, 15, DARKBLUE);
    
    DrawText(TextFormat("Steering: %.2f", car.steeringAngle), col3X, metricsY + 40, 15, DARKBLUE);
    
    // Status line
    Color statusColor = paused ? ORANGE : DARKGREEN;
    DrawText(TextFormat("Simulation Speed: %.1fx | %s", 
                      simulationSpeed, paused ? "PAUSED" : "RUNNING"), 
            x + 10, metricsY + 60, 15, statusColor);
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

void VisualizationHelper::drawLaneDeviationGraph(int x, int y, int width, int height, float laneWidth) {
    // Semi-transparent background
    DrawRectangle(x, y, width, height, Fade(LIGHTGRAY, 0.7f));
    
    // Title
    DrawText("Lane Deviation", x + 10, y + 5, 15, DARKBLUE);
    
    // Draw history graph
    if (lateralDeviationHistory.size() > 1) {
        int graphX = x + 10;
        int graphY = y + 25;
        int graphWidth = width - 20;
        int graphHeight = height - 35;
        
        // Background
        DrawRectangle(graphX, graphY, graphWidth, graphHeight, Fade(WHITE, 0.9f));
        
        float halfHeight = graphHeight / 2.0f;
        float centerY = graphY + halfHeight;
        
        // Draw lane center line
        DrawLine(
            graphX, centerY,
            graphX + graphWidth, centerY,
            Fade(DARKGREEN, 0.7f)
        );
        
        // Draw lane edges
        float laneHalfWidth = laneWidth / 2.0f;
        float yScale = halfHeight / (laneHalfWidth * 1.5f);  // Show 1.5x lane width
        
        DrawLine(
            graphX, centerY - (laneHalfWidth * yScale),
            graphX + graphWidth, centerY - (laneHalfWidth * yScale),
            Fade(ORANGE, 0.7f)
        );
        
        DrawLine(
            graphX, centerY + (laneHalfWidth * yScale),
            graphX + graphWidth, centerY + (laneHalfWidth * yScale),
            Fade(ORANGE, 0.7f)
        );
        
        // Draw deviation history
        float xScale = (float)graphWidth / (historySize - 1);
        
        for (size_t i = 0; i < lateralDeviationHistory.size() - 1; i++) {
            float x1 = graphX + i * xScale;
            float y1 = centerY - (lateralDeviationHistory[i] * yScale);
            float x2 = graphX + (i + 1) * xScale;
            float y2 = centerY - (lateralDeviationHistory[i + 1] * yScale);
            
            DrawLine(x1, y1, x2, y2, Fade(BLUE, 0.8f));
        }
        
        // Draw current deviation label
        if (!lateralDeviationHistory.empty()) {
            DrawText(TextFormat("%.2f m", lateralDeviationHistory.back()), 
                    graphX + graphWidth - 60, graphY + 5, 15, BLUE);
        }
    }
}

void VisualizationHelper::drawMinimalHelp(int x, int y, const char* helpText) {
    DrawText(helpText, x, y, 15, Fade(DARKGRAY, 0.7f));
}

std::string VisualizationHelper::getActionName(DQNEnvironment::Action action) {
    // Steering group
    int steeringGroup = action / 9;
    std::string steeringText;
    
    switch(steeringGroup) {
        case 0: steeringText = "NEUTRAL"; break;
        case 1: steeringText = "LEFT"; break;
        case 2: steeringText = "RIGHT"; break;
        default: steeringText = "UNKNOWN";
    }
    
    // Throttle/brake action
    int throttleBrakeAction = action % 9;
    std::string controlText;
    
    switch(throttleBrakeAction) {
        case 0: controlText = "STRONG BRAKE"; break;
        case 1: controlText = "MEDIUM BRAKE"; break;
        case 2: controlText = "LIGHT BRAKE"; break;
        case 3: controlText = "COAST"; break;
        case 4: controlText = "LIGHT THROTTLE"; break;
        case 5: controlText = "MEDIUM THROTTLE"; break;
        case 6: controlText = "STRONG THROTTLE"; break;
        case 7: controlText = "FULL THROTTLE"; break;
        case 8: controlText = "NO CHANGE"; break;
        default: controlText = "UNKNOWN";
    }
    
    return controlText + " " + steeringText;
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

Color VisualizationHelper::getLaneDeviationColor(float deviation, float laneWidth) {
    // Calculate color based on how close we are to the lane center
    float halfLaneWidth = laneWidth / 2.0f;
    float ratio = std::abs(deviation) / halfLaneWidth;
    
    if (ratio < 0.5f) {
        // Close to center - green
        unsigned char g = 200 + static_cast<unsigned char>(55 * (1.0f - ratio / 0.5f));
        return {0, g, 0, 255};
    } else if (ratio <= 1.0f) {
        // Getting close to edge - yellow
        float adjustment = (ratio - 0.5f) / 0.5f;
        unsigned char r = static_cast<unsigned char>(255 * adjustment);
        unsigned char g = static_cast<unsigned char>(255 * (1.0f - adjustment) + 200 * adjustment);
        return {r, g, 0, 255};
    } else {
        // Outside lane - red
        float excess = std::min(ratio - 1.0f, 1.0f);
        unsigned char r = 255;
        unsigned char g = static_cast<unsigned char>(200 * (1.0f - excess));
        return {r, g, 0, 255};
    }
}

} // namespace CarGame