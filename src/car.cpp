#include "../include/car.h"

// Car constants definitions
const float carWidth = 2.0f;  // Width of the car
const float carLength = 4.0f; // Length of the car
const float wheelBase = 2.8f;
const float l_r = 1.2f;
const float l_f = 1.6f;
const float carHeight = 1.0f; // Height of the car
const float steeringSpeedConstant = 1.0f;

// Updates the car's speed, rotation, and position based on keyboard input.
void UpdateCar(Car *car, float deltaTime) {
    HandleHorizontalMovement(car, deltaTime); // Handle acceleration and deceleration
    HandleLateralMovement(car, deltaTime); // Handle steering and lateral movement
}

void HandleHorizontalMovement(Car *car, float deltaTime) {
    // Accelerate or decelerate the car
    if (IsKeyDown(KEY_UP)) {
        car->speed += 5.0f * deltaTime;
        if (car->speed > 10.0f) car->speed = 10.0f;
    } else if (IsKeyDown(KEY_DOWN)) {
        car->speed -= 5.0f * deltaTime;
        if (car->speed < -5.0f) car->speed = -5.0f;
    } else {
        // Apply friction to gradually reduce speed
        if (car->speed > 0.0f) {
            car->speed -= 2.0f * deltaTime;
            if (car->speed < 0.0f) car->speed = 0.0f;
        } else if (car->speed < 0.0f) {
            car->speed += 2.0f * deltaTime;
            if (car->speed > 0.0f) car->speed = 0.0f;
        }
    }
}

void HandleLateralMovement(Car *car, float deltaTime) {
    // Steering: only allow steering when the car is moving
    if (fabs(car->speed) > 0.1f) {
        if (IsKeyDown(KEY_LEFT)) {
            car->steeringSpeed = steeringSpeedConstant * deltaTime;
            car->steeringAngle += car->steeringSpeed;
            if (car->steeringAngle > 0.5f) car->steeringAngle = 0.5f;
        }
        else  if (IsKeyDown(KEY_RIGHT)) {
            car->steeringSpeed = -steeringSpeedConstant * deltaTime;
            car->steeringAngle += car->steeringSpeed;
            if (car->steeringAngle < -0.5f) car->steeringAngle = -0.5f;
        } else {
            // Gradually reduce steering angle when no input
            if (car->steeringAngle > 0.0f) {
                car->steeringAngle -= 1.0f * deltaTime;
                if (car->steeringAngle < 0.0f) car->steeringAngle = 0.0f;
            } else if (car->steeringAngle < 0.0f) {
                car->steeringAngle += 1.0f * deltaTime;
                if (car->steeringAngle > 0.0f) car->steeringAngle = 0.0f;
            }
        }     
    }

    float beta = atan2f((l_r) * tanf(car->steeringAngle), (l_f + l_r));

    float omega_dot = car->speed * cosf(beta) * tanf(car->steeringAngle) / wheelBase;
    car->rotation += omega_dot * deltaTime; // Update rotation based on steering angle
    if (car->rotation > 2 * PI) car->rotation -= 2 * PI;
    if (car->rotation < 0) car->rotation += 2 * PI; 

    // Update car's velocity based on speed and rotation
    car->velocity.z = car->speed * cosf(car->rotation + beta);
    car->velocity.x = car->speed * sinf(car->rotation + beta);

    // Update car's position based on velocity
    car->position.x += car->velocity.x * deltaTime;
    car->position.z += car->velocity.z * deltaTime;

    // Keep the car above the ground
    if (car->position.y < 0.5f) {
        car->position.y = 0.5f;
    }
}