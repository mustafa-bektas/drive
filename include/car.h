// car.h
#ifndef CAR_H
#define CAR_H

#include "raylib.h"
#include <math.h>

// Car physics constants
typedef struct {
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
    float minMovementSpeed;  // Minimum speed for steering to work
    int gearRatio;
    float tireRadius;
} CarPhysicsConfig;

// Car structure
typedef struct {
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
    float speed;
    float rotation;  // in radians
    float steeringAngle; // in radians
    float steeringSpeed; // in radians per second
    CarPhysicsConfig config;
} Car;

// Define the RAD2DEG macro if it doesn't exist
#ifndef RAD2DEG
    #define RAD2DEG (180.0f/PI)
#endif

// Car initialization
Car CreateCar(Vector3 startPosition);

// Function prototypes for car movement
void UpdateCar(Car *car, float deltaTime);
void HandleLongitudinalMovement(Car *car, float deltaTime);
void HandleLateralMovement(Car *car, float deltaTime);

// Helper functions for movement
void ProcessSteeringInput(Car *car, float deltaTime);
void CalculateSteering(Car *car, float deltaTime, float *beta);
void UpdateCarPosition(Car *car, float deltaTime, float beta);
void ApplyFriction(Car *car, float deltaTime);
void ReturnSteeringToCenter(Car *car, float deltaTime);
void NormalizeRotation(Car *car);

#endif // CAR_H