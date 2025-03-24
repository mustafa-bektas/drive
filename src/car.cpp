#include "../include/car.h"
#include "car.h"
#include <utility>
#include <cstdio>

namespace CarGame {

CarPhysicsConfig::CarPhysicsConfig(
    float width,
    float length,
    float height,
    float wheelBase,
    float frontAxleDistance,
    float rearAxleDistance,
    float minSpeed,
    float maxSteeringAngle,
    float steeringSpeed,
    float minMovementSpeed,
    int gearRatio,
    float tireRadius,
    float inertiaAtEngine,
    int mass)
    : width(width), 
      length(length), 
      height(height),
      wheelBase(wheelBase),
      frontAxleDistance(frontAxleDistance),
      rearAxleDistance(rearAxleDistance),
      minSpeed(minSpeed),
      maxSteeringAngle(maxSteeringAngle),
      steeringSpeed(steeringSpeed),
      minMovementSpeed(minMovementSpeed),
      gearRatio(gearRatio),
      tireRadius(tireRadius),
      inertiaAtEngine(inertiaAtEngine),
      mass(mass) {
}

Car::Car(const Vector3& startPosition) 
    : position(startPosition),
      velocity({0.0f, 0.0f, 0.0f}),
      acceleration({0.0f, 0.0f, 0.0f}),
      speed(0.0f),
      rotation(0.0f),
      steeringAngle(0.0f),
      steeringSpeed(0.0f),
      config(),  // Uses default values from constructor
      engineSpeed(0.0f),
      engineSpeed_dot(0.0f),
      throttle(0.0f),
      brake(0.0f),
      slipRatio(0.0f),
      longitudinalForce(0.0f),
      dragForce(0.0f),
      rollingResistance(0.0f),
      wheelRotationSpeed(0.0f),
      clutch(false),
      netForce(0.0f) {
}

// Updates the car's physics
void Car::update(float deltaTime) {
    updateLongitudinalPhysics(deltaTime);
    updateLateralPhysics(deltaTime);
}

void Car::updateLongitudinalPhysics(float deltaTime) {
    // Update engine speed based on throttle input
    // Engine idle speed
    const float idleRPM = 1.0f;
    
    if (throttle > 0.1f) {
        // Engine revs up when throttle is applied (even when stationary)
        /* float maxRPM = 8000.0f;
        float targetRPM = idleRPM + throttle * (maxRPM - idleRPM);
        engineSpeed += (targetRPM - engineSpeed) * deltaTime * 3.0f; */
        clutch = false; // Clutch disengaged
        engineSpeed_dot = getEngineTorque(throttle, engineSpeed) / config.inertiaAtEngine;
        engineSpeed += engineSpeed_dot * deltaTime;
        if (engineSpeed > 8000.0f) engineSpeed = 8000.0f; // Limit engine speed
    } else if (std::abs(speed) < 0.5f) {
        // At idle when stopped
        engineSpeed = idleRPM;
        clutch = true; // Clutch engaged
    } else {
        // When moving, match engine speed to wheel rotation
        float wheelRPM = std::abs(speed) / config.tireRadius * config.gearRatio * (60.0f / (2.0f * PI));
        engineSpeed = std::max(idleRPM, wheelRPM);
    }
    
    // Limit engine speed
    if (engineSpeed > 8000.0f) engineSpeed = 8000.0f;
    
    // Calculate engine torque based on updated engine speed
    float engineTorque = getEngineTorque(throttle, engineSpeed);
    
    if (throttle < 0.1f && engineSpeed > 1000.0f) {
        engineTorque -= (engineSpeed / 8000.0f) * 75.0f; // Engine braking increases with RPM
    }

    wheelRotationSpeed = clutch ? wheelRotationSpeed : engineSpeed / config.gearRatio * (2.0f * PI / 60.0f);

    float brakeTorque = 0.0f;
    if (brake > 0.0f) {
        // Up to 5000 N·m braking torque
        brakeTorque = brake * 8000.0f; 
        
        // Brakes always act to slow rotation toward zero
        if (wheelRotationSpeed > 0.0f) {
            brakeTorque = -brakeTorque;
        } else if (wheelRotationSpeed < 0.0f) {
            // Already correct direction
        } else {
            // When wheels aren't rotating
            brakeTorque = 0.0f;
        }
    }

    float wheelInertia = 5.0f;  // kg·m²
    float wheelAngularAccel = (engineTorque + brakeTorque) / wheelInertia;
    
    // STEP 7: Update wheel rotation speed based on torques
    wheelRotationSpeed += wheelAngularAccel * deltaTime;
    
    // STEP 8: Calculate tire slip using actual wheel rotation
    float wheelLinearSpeed = wheelRotationSpeed * config.tireRadius;

    slipRatio = calculateSlipRatio(wheelLinearSpeed, speed);    
    
    // Longitudinal force is limited by tire grip (Pacejka model)
    longitudinalForce = 2 * calculateTireForcePacejka(slipRatio); // For 2 drive wheels
    
    // Calculate total resistance
    float totalResistance = getTotalResistanceForces(deltaTime);
    
    // Net force on the vehicle
    netForce = longitudinalForce - totalResistance;

    // Calculate acceleration (F = ma)
    acceleration.x = netForce / config.mass;
    
    // Update speed
    speed += acceleration.x * deltaTime;
}

float Car::getTotalResistanceForces(float deltaTime) {
    // Air density (kg/m³)
    const float airDensity = 1.225f;
    
    // Drag coefficient (dimensionless)
    const float dragCoefficient = 0.3f;
    
    // Frontal area (m²)
    float frontalArea = config.width * config.height * 0.8f; // Not all of width*height is effective
    
    // Drag force: 0.5 * ρ * Cd * A * v²
    dragForce = 0.5f * airDensity * dragCoefficient * frontalArea * speed * speed;
    
    // Rolling resistance coefficient (increases slightly with speed)
    float rollingCoefficient = 0.015f * (1.0f + std::abs(speed) * 0.01f);
    //float rollingCoefficient = 0.0f;
    
    // Rolling resistance force: Cr * m * g
    rollingResistance = rollingCoefficient * config.mass * 9.81f;
    
    // Add direction to rolling resistance
    if (speed != 0.0f) {
        rollingResistance *= (speed > 0.0f ? 1.0f : -1.0f);
    }
    
    return dragForce + rollingResistance;
}



float Car::calculateTireForcePacejka(float slipRatio) const {
    // Pacejka parameters
    float D = 1.0f;     // Peak coefficient (dimensionless)
    float C = 1.5f;     // Shape factor
    float B = 10.0f;    // Stiffness factor
    float E = 0.1f;     // Curvature factor
    float Fz = 4000.0f; // Normal load per tire (N)

    // Calculate coefficient from Magic Formula
    float coefficient = D * std::sinf(C * std::atanf(B * slipRatio - E * (B * slipRatio - std::atanf(B * slipRatio))));
    
    // Apply coefficient to normal load
    return coefficient * Fz;
}

float Car::calculateSlipRatio(float wheelLinearSpeed, float vehicleSpeed) const {
    // Calculate absolute vehicle speed for denominator calculation
    float speedAbs = std::abs(vehicleSpeed);
    const float minSpeed = 0.5f;
    float result;
    
    // Handle low-speed scenario to avoid division by near-zero
    if (speedAbs < minSpeed) {
        // During near-standstill, use minimum speed as denominator
        // This creates a smooth transition and avoids numerical instability
        result = (wheelLinearSpeed - vehicleSpeed) / minSpeed;
    } else {
        // Normal driving - proper slip ratio calculation
        // Using absolute speed in denominator preserves slip direction
        result = (wheelLinearSpeed - vehicleSpeed) / speedAbs;
    }
    
    // Clamp to physically reasonable values (-1 to 1)
    // Values outside this range are theoretically possible but rarely useful in simulation
    result = std::max(-1.0f, std::min(result, 1.0f));
    return result;
}

float Car::getEngineTorque(float throttle, float rpm) const {
    return throttle * (400.0f + 250.0f * (rpm / 4000.0f) * (1.0f - rpm / 8000.0f));
}

void Car::updateLateralPhysics(float deltaTime) {
    // Calculate steering effects
    float beta = 0.0f;
    
    // Only apply steering if car is moving fast enough
    if (std::fabs(speed) > config.minMovementSpeed) {
        // Calculate slip angle
        beta = std::atan2f((config.rearAxleDistance) * std::tanf(steeringAngle), 
                     (config.frontAxleDistance + config.rearAxleDistance));

        // Calculate rotation rate
        float omega_dot = speed * std::cosf(beta) * std::tanf(steeringAngle) / config.wheelBase;
        
        // Update rotation and normalize to 0-2PI range
        rotation += omega_dot * deltaTime;
        normalizeRotation();
    }
    
    // Update car's velocity based on speed and rotation
    velocity.z = speed * std::cosf(rotation + beta);
    velocity.x = speed * std::sinf(rotation + beta);

    // Update car's position based on velocity
    position.x += velocity.x * deltaTime;
    position.z += velocity.z * deltaTime;

    // Keep the car above the ground
    if (position.y < 0.5f) {
        position.y = 0.5f;
    }
}

void Car::normalizeRotation() {
    if (rotation > 2 * PI) rotation -= 2 * PI;
    if (rotation < 0) rotation += 2 * PI;
}
} // namespace CarGame