#pragma once

#include "raylib.h"
#include "car.h"

namespace CarGame {

/**
 * Camera class that handles camera movement and tracking
 */
class GameCamera {
public:
    GameCamera();
    
    // Update camera to follow the car
    void update(const Car& car);
    
    const Camera& getCamera() const { return camera; }
    
private:
    Camera camera;
};

} // namespace CarGame