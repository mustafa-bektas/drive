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
        float mass = 1600,
        float corneringStiffnessFront = 50000.0f,
        float corneringStiffnessRear = 40000.0f,
        float pacejkaB_lat = 10.0f,  
        float pacejkaC_lat = 1.5f,    
        float pacejkaD_lat = 1.0f,   
        float pacejkaE_lat = -0.5,   
        float frontWeight = 0.55f,    
        float normalLoadFront = 4000.0f, 
        float normalLoadRear = 3800.0f,  
        float heightCG = 0.5f         
    );

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
    float mass;
    float corneringStiffnessFront; // Cornering stiffness for front tires
    float corneringStiffnessRear;  // Cornering stiffness for rear tires
    float pacejkaB_lat;            // Pacejka B parameter for lateral forces
    float pacejkaC_lat;            // Pacejka C parameter for lateral forces
    float pacejkaD_lat;            // Pacejka D parameter for lateral forces
    float pacejkaE_lat;            // Pacejka E parameter for lateral forces
    float frontWeight;             // Weight distribution towards the front
    float normalLoadFront;         // Normal load on front tires
    float normalLoadRear;          // Normal load on rear tires
    float heightCG;               // Height of the center of gravity
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
    float calculatePacejkaLateral(float slipAngle, float Fz, bool isFrontTire) const;

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

    // Lateral Dynamics
    float slipAngleFront;  // Slip angle of front tires
    float slipAngleRear;   // Slip angle of rear tires
    float lateralForceFront; // Lateral force on front tires
    float lateralForceRear;  // Lateral force on rear tires
    float yawMoment;       // Yaw moment around center of gravity
    float yawRate;         // Yaw rate (angular velocity)
    float lateralVelocity; // Lateral velocity in car's reference frame
    float sideSlipAngle;
    float sideSlipAngleDot;
    float yawAngle;
    float yawAngleDot;
    float lateralAcceleration;     // Lateral acceleration (m/s²)
    float yawAngleDotDot;
};

} // namespace CarGame