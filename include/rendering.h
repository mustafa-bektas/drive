#pragma once

#include "raylib.h"
#include "car.h"
#include "camera.h"
#include <vector>
#include <functional>

namespace CarGame {

/**
 * Renderer class to handle all rendering operations
 * Responsible for drawing the 3D scene and UI information
 */
class Renderer {
public:
    // Constructor and destructor
    Renderer();
    ~Renderer();
    
    // Initialize renderer with car properties
    void initialize(const Car& car);
    
    // Main rendering method
    void drawScene(const GameCamera& camera, const Car& car, const Vector3& floorPosition);
    
private:
    // 3D rendering methods
    void draw3DScene(const GameCamera& camera, const Car& car, const Vector3& floorPosition);
    
    // UI rendering methods
    void drawTelemetryPanel(const Car& car);
    void drawInstructions();
    
    // Helper method for drawing UI sections with modern C++ approach
    int drawSection(int x, int y, const char* title, 
                   const std::vector<std::function<void(int, int)>>& drawFuncs);
    
    // Assets
    Model carModel;
};

} // namespace CarGame