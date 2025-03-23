#pragma once

#include "raylib.h"
#include "car.h"
#include "camera.h"

namespace CarGame {

/**
 * Renderer class to handle all rendering operations
 */
class Renderer {
public:
    Renderer();
    ~Renderer();
    
    void initialize(const Car& car);
    
    // Draw the 3D scene and UI
    void drawScene(const GameCamera& camera, const Car& car, const Vector3& floorPosition);
    
private:
    Model carModel;
};

} // namespace CarGame