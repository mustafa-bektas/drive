#pragma once

#include "raylib.h"
#include <cmath>

namespace CarGame {

class Car;

/**
 * Car physics configuration class
 */
class CarPhysicsConfig {
public:
    // Constructor with sensible defaults
    CarPhysicsConfig(
        float width = 2.0f,
        float length = 4.0f,
        float height = 1.0f,
        float wheelBase = 2.8f,
        float frontAxleDistance = 1.6f,
        float rearAxleDistance = 1.2f,
        float maxSpeed = 10.0f,
        float minSpeed = -5.0f,
        float maxSteeringAngle = 0.5f,
        float steeringSpeed = 1.0f,
        float frictionForce = 2.0f,
        float accelerationForce = 5.0f,
        float minMovementSpeed = 0.1f,
        int gearRatio = 3,
        float tireRadius = 0.35f,
        float inertiaAtEngine = 0.5f);

    float width;
    float length;
    float height;
    float wheelBase;
    float frontAxleDistance;
    float rearAxleDistance;
    float maxSpeed;
    float minSpeed;
    float maxSteeringAngle;
    float steeringSpeed;
    float frictionForce;
    float accelerationForce;
    float minMovementSpeed;
    int gearRatio;
    float tireRadius;
    float inertiaAtEngine;
};


class Car {
public:
    explicit Car(const Vector3& startPosition);

    // Main update method
    void update(float deltaTime);
    
    // Movement handling
    void handleLongitudinalMovement(float deltaTime);
    void handleLateralMovement(float deltaTime);
    
    // Helper methods
    void processSteeringInput(float deltaTime);
    void calculateSteering(float deltaTime, float& beta);
    void updatePosition(float deltaTime, float beta);
    void applyFriction(float deltaTime);
    void returnSteeringToCenter(float deltaTime);
    void normalizeRotation();
    float getEngineTorque(float throttle, float rpm) const;

    // Car state
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
    float speed;
    float rotation;  // in radians
    float steeringAngle; // in radians
    float steeringSpeed; // in radians per second
    CarPhysicsConfig config;
    float engineSpeed;
    float engineSpeed_dot;
    float throttle; // Throttle position (0.0 to 1.0)
    float brake;    // Brake position (0.0 to 1.0)
};

} // namespace CarGame