#pragma once

#include "raylib.h"
#include <cmath>

namespace CarGame {

class Car;

class CarPhysicsConfig {
public:
    // Constructor with sensible defaults
    CarPhysicsConfig(
        float width = 1.8f,
        float length = 4.5f,
        float height = 1.5f,
        float wheelBase = 2.8f,
        float frontAxleDistance = 1.6f,
        float rearAxleDistance = 1.2f,
        float minSpeed = -1.0f,
        float maxSteeringAngle = 0.6f,
        float steeringSpeed = 2.0f,
        float minMovementSpeed = 0.5f,
        int gearRatio = 7,
        float tireRadius = 0.33f,
        float inertiaAtEngine = 0.45f,
        int mass = 1600);

    float width;
    float length;
    float height;
    float wheelBase;
    float frontAxleDistance;
    float rearAxleDistance;
    float minSpeed;
    float maxSteeringAngle;
    float steeringSpeed;
    float minMovementSpeed;
    int gearRatio;
    float tireRadius;
    float inertiaAtEngine;
    int mass;
};


class Car {
public:
    explicit Car(const Vector3& startPosition);

    // Main update method
    void update(float deltaTime);
    
    // Physics methods
    float getTotalResistanceForces(float deltaTime);
    void updateLongitudinalPhysics(float deltaTime);
    void updateLateralPhysics(float deltaTime);
    float getEngineTorque(float throttle, float rpm) const;
    void normalizeRotation();
    float calculateTireForcePacejka(float slipRatio) const;
    float calculateSlipRatio(float wheelLinearSpeed, float vehicleSpeed) const;

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
    float slipRatio;
    float longitudinalForce;
    float dragForce;
    float rollingResistance;
    float wheelRotationSpeed; // in radians per second
    bool clutch;
    float netForce;
};

} // namespace CarGame