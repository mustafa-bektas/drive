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

    float getWidth() const { return width; }
    float getLength() const { return length; }
    float getHeight() const { return height; }
    float getWheelBase() const { return wheelBase; }
    float getFrontAxleDistance() const { return frontAxleDistance; }
    float getRearAxleDistance() const { return rearAxleDistance; }
    float getMaxSpeed() const { return maxSpeed; }
    float getMinSpeed() const { return minSpeed; }
    float getMaxSteeringAngle() const { return maxSteeringAngle; }
    float getSteeringSpeed() const { return steeringSpeed; }
    float getFrictionForce() const { return frictionForce; }
    float getAccelerationForce() const { return accelerationForce; }
    float getMinMovementSpeed() const { return minMovementSpeed; }
    int getGearRatio() const { return gearRatio; }
    float getTireRadius() const { return tireRadius; }
    float getInertiaAtEngine() const { return inertiaAtEngine; }

private:
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
    // Constructor
    explicit Car(const Vector3& startPosition);

    void update(float deltaTime);
    
    const Vector3& getPosition() const { return position; }
    const Vector3& getVelocity() const { return velocity; }
    const Vector3& getAcceleration() const { return acceleration; }
    float getSpeed() const { return speed; }
    float getRotation() const { return rotation; }
    float getSteeringAngle() const { return steeringAngle; }
    float getThrottle() const { return throttle; }
    float getBrake() const { return brake; }
    const CarPhysicsConfig& getConfig() const { return config; }

private:
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