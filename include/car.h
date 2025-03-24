#pragma once

#include "raylib.h"
#include <cmath>

namespace CarGame {

class Car;

class CarPhysicsConfig {
public:
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
        float pacejkaC_lat = 1.3f,    
        float pacejkaD_lat = 1.0f,   
        float pacejkaE_lat = 1,   
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
    float corneringStiffnessFront;
    float corneringStiffnessRear;
    float pacejkaB_lat;
    float pacejkaC_lat;
    float pacejkaD_lat;
    float pacejkaE_lat;
    float frontWeight;
    float normalLoadFront;
    float normalLoadRear;
    float heightCG;
};

class Car {
public:
    explicit Car(const Vector3& startPosition);

    void update(float deltaTime);
    
    float getTotalResistanceForces(float deltaTime);
    void updateLongitudinalPhysics(float deltaTime);
    void updateLateralPhysics(float deltaTime);
    float getEngineTorque(float throttle, float rpm) const;
    void normalizeRotation();
    float calculateTireForcePacejka(float slipRatio) const;
    float calculateSlipRatio(float wheelLinearSpeed, float vehicleSpeed) const;
    float calculatePacejkaLateral(float slipAngle, float Fz, bool isFrontTire) const;

    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
    float speed;
    float rotation;
    float steeringAngle;
    float steeringSpeed;
    CarPhysicsConfig config;
    float engineSpeed;
    float engineSpeed_dot;
    float throttle;
    float brake;
    float slipRatio;
    float longitudinalForce;
    float dragForce;
    float rollingResistance;
    float wheelRotationSpeed;
    bool clutch;
    float netForce;

    float slipAngleFront;
    float slipAngleRear;
    float lateralForceFront;
    float lateralForceRear;
    float yawMoment;
    float yawRate;
    float lateralVelocity;
    float sideSlipAngle;
    float sideSlipAngleDot;
    float yawAngle;
    float yawAngleDot;
    float lateralAcceleration;
    float yawAngleDotDot;
};

} // namespace CarGame