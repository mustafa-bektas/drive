#pragma once

#include "car.h"

namespace CarGame {

class InputHandler {
public:
    InputHandler() = default;
    
    void processInput(Car& car, float deltaTime);
    
private:
    void processThrottleAndBrake(Car& car, float deltaTime);
    void processSteering(Car& car, float deltaTime);
};

} // namespace CarGame