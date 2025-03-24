#include "../include/input_handler.h"
#include "raylib.h"

namespace CarGame {

void InputHandler::processInput(Car& car, float deltaTime) {
    processThrottleAndBrake(car, deltaTime);
    processSteering(car, deltaTime);
}

void InputHandler::processThrottleAndBrake(Car& car, float deltaTime) {
    // Handle throttle and brake input
    if (IsKeyDown(KEY_UP)) {
        car.throttle += 3.0f * deltaTime;
        if (car.throttle > 1.0f) car.throttle = 1.0f; 
    } else if (IsKeyDown(KEY_DOWN)) {
        car.brake += 3.0f * deltaTime;
        if (car.brake > 1.0f) car.brake = 1.0f;
    }
    else {
        car.throttle = 0.0f;
        car.brake = 0.0f;
    }
}

void InputHandler::processSteering(Car& car, float deltaTime) {
    // Only process steering if car is moving fast enough
    if (std::fabs(car.speed) > car.config.minMovementSpeed) {
        if (IsKeyDown(KEY_LEFT)) {
            car.steeringSpeed = car.config.steeringSpeed * deltaTime;
            car.steeringAngle += car.steeringSpeed;
            if (car.steeringAngle > car.config.maxSteeringAngle) 
                car.steeringAngle = car.config.maxSteeringAngle;
        }
        else if (IsKeyDown(KEY_RIGHT)) {
            car.steeringSpeed = -car.config.steeringSpeed * deltaTime;
            car.steeringAngle += car.steeringSpeed;
            if (car.steeringAngle < -car.config.maxSteeringAngle) 
                car.steeringAngle = -car.config.maxSteeringAngle;
        } else {
            // Gradually reduce steering angle when no input
            float returnSpeed = car.config.steeringSpeed * deltaTime;
            
            if (car.steeringAngle > 0.0f) {
                car.steeringAngle -= returnSpeed;
                if (car.steeringAngle < 0.0f) car.steeringAngle = 0.0f;
            } else if (car.steeringAngle < 0.0f) {
                car.steeringAngle += returnSpeed;
                if (car.steeringAngle > 0.0f) car.steeringAngle = 0.0f;
            }
        }
    }
}

} // namespace CarGame