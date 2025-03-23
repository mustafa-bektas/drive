#pragma once

#include "car.h"

namespace CarGame {

class InputHandler {
public:
    // Constructor
    InputHandler() = default;
    
    // Process input for the current frame
    void processInput(Car& car, float deltaTime);
    
private:
    // Helper methods
    void processThrottleAndBrake(Car& car, float deltaTime);
    void processSteering(Car& car, float deltaTime);
};

} // namespace CarGame