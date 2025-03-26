#pragma once

#include "raylib.h"
#include "car.h"

namespace CarGame {

// camera follows the car
class GameCamera {
public:
    GameCamera();
    
    // update cam pos based on car
    void update(const Car& car);
    
    const Camera& getCamera() const { return camera; }
    
private:
    Camera camera;
};

} // namespace CarGame