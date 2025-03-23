#include "../include/car.h"

// Create car with default physics configuration
Car CreateCar(Vector3 startPosition) {
    CarPhysicsConfig config = {
        2.0f,   // width
        4.0f,   // length
        1.0f,   // height
        2.8f,   // wheelBase
        1.6f,   // frontAxleDistance
        1.2f,   // rearAxleDistance
        10.0f,  // maxSpeed
        -5.0f,  // minSpeed
        0.5f,   // maxSteeringAngle
        1.0f,   // steeringSpeed
        2.0f,   // frictionForce
        5.0f,   // accelerationForce
        0.1f,   // minMovementSpeed
        3,      // gearRatio
        0.35f,  // tireRadius
        0.5f   // inertiaAtEngine
    };
    
    Car car = {
        startPosition,            // position
        {0.0f, 0.0f, 0.0f},       // velocity
        {0.0f, 0.0f, 0.0f},       // acceleration
        0.0f,                     // speed
        0.0f,                     // rotation
        0.0f,                     // steeringAngle
        0.0f,                     // steeringSpeed
        config,                   // physics configuration
        0.0f,                     // engineSpeed
        0.0f,                     // engineSpeed_dot
        0.0f,                     // throttle
        0.0f                      // brake
    };
    
    return car;
}

// Updates the car's speed, rotation, and position based on keyboard input.
void UpdateCar(Car *car, float deltaTime) {
    HandleLongitudinalMovement(car, deltaTime); // Handle acceleration and deceleration
    HandleLateralMovement(car, deltaTime); // Handle steering and lateral movement
}

void HandleLongitudinalMovement(Car *car, float deltaTime) {
    // Handle throttle and brake input
    if (IsKeyDown(KEY_UP)) {
        car->throttle += 3.0f * deltaTime;
        if (car->throttle > 1.0f) car->throttle = 1.0f; 
    } else if (IsKeyDown(KEY_DOWN)) {
        car->brake += 3.0f * deltaTime;
        if (car->brake > 1.0f) car->brake = 1.0f;
    }
    else {
        // Gradually reduce throttle and brake when no input
        car->throttle -= 5.0f * deltaTime;
        car->brake -= 5.0f * deltaTime;
        if (car->throttle < 0.0f) car->throttle = 0.0f;
        if (car->brake < 0.0f) car->brake = 0.0f;
    }
}

float GetEngineTorque(float throttle, float rpm)
{
    return throttle * (-0.0003f * rpm * rpm + 0.1f * rpm + 500.0f);
}

void ApplyFriction(Car *car, float deltaTime) {
    if (car->speed > 0.0f) {
        car->speed -= car->config.frictionForce * deltaTime;
        if (car->speed < 0.0f) car->speed = 0.0f;
    } else if (car->speed < 0.0f) {
        car->speed += car->config.frictionForce * deltaTime;
        if (car->speed > 0.0f) car->speed = 0.0f;
    }
}

void HandleLateralMovement(Car *car, float deltaTime) {
    float beta = 0.0f;
    
    // Only process steering if car is moving fast enough
    if (fabs(car->speed) > car->config.minMovementSpeed) {
        ProcessSteeringInput(car, deltaTime);
        CalculateSteering(car, deltaTime, &beta);
    }
    
    UpdateCarPosition(car, deltaTime, beta);
}

void ProcessSteeringInput(Car *car, float deltaTime) {
    if (IsKeyDown(KEY_LEFT)) {
        car->steeringSpeed = car->config.steeringSpeed * deltaTime;
        car->steeringAngle += car->steeringSpeed;
        if (car->steeringAngle > car->config.maxSteeringAngle) 
            car->steeringAngle = car->config.maxSteeringAngle;
    }
    else if (IsKeyDown(KEY_RIGHT)) {
        car->steeringSpeed = -car->config.steeringSpeed * deltaTime;
        car->steeringAngle += car->steeringSpeed;
        if (car->steeringAngle < -car->config.maxSteeringAngle) 
            car->steeringAngle = -car->config.maxSteeringAngle;
    } else {
        // Gradually reduce steering angle when no input
        ReturnSteeringToCenter(car, deltaTime);
    }
}

void ReturnSteeringToCenter(Car *car, float deltaTime) {
    float returnSpeed = car->config.steeringSpeed * deltaTime;
    
    if (car->steeringAngle > 0.0f) {
        car->steeringAngle -= returnSpeed;
        if (car->steeringAngle < 0.0f) car->steeringAngle = 0.0f;
    } else if (car->steeringAngle < 0.0f) {
        car->steeringAngle += returnSpeed;
        if (car->steeringAngle > 0.0f) car->steeringAngle = 0.0f;
    }
}

void CalculateSteering(Car *car, float deltaTime, float *beta) {
    // Calculate slip angle
    *beta = atan2f((car->config.rearAxleDistance) * tanf(car->steeringAngle), 
                  (car->config.frontAxleDistance + car->config.rearAxleDistance));

    // Calculate rotation rate
    float omega_dot = car->speed * cosf(*beta) * tanf(car->steeringAngle) / car->config.wheelBase;
    
    // Update rotation and normalize to 0-2PI range
    car->rotation += omega_dot * deltaTime;
    NormalizeRotation(car);
}

void NormalizeRotation(Car *car) {
    if (car->rotation > 2 * PI) car->rotation -= 2 * PI;
    if (car->rotation < 0) car->rotation += 2 * PI;
}

void UpdateCarPosition(Car *car, float deltaTime, float beta) {
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