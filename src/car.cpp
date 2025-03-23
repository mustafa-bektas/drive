#include "../include/car.h"
#include "car.h"
#include <utility>

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
      rollingResistance(0.0f) {
}

// Updates the car's physics
void Car::update(float deltaTime) {
    updateLongitudinalPhysics(deltaTime);
    updateLateralPhysics(deltaTime);
}

void Car::updateLongitudinalPhysics(float deltaTime) {
    float engineTorque = 0.00f;
    if (brake < 0.5f) {
        engineTorque = getEngineTorque(throttle, engineSpeed);
    }
    
    float totalLoadTorque = config.gearRatio * config.tireRadius * getTotalResistanceForces(deltaTime);
    engineSpeed_dot = (engineTorque - totalLoadTorque) / config.inertiaAtEngine;
    engineSpeed += engineSpeed_dot * deltaTime;
    if (engineSpeed < 0.0f) engineSpeed = 0.0f; // Prevent negative engine speed
    if (engineSpeed > 7000.0f) engineSpeed = 7000.0f; // Prevent excessive engine speed
    float wheelSpeed = engineSpeed / config.gearRatio;

    // calculate longitudinal slip ratio
    slipRatio = calculateSlipRatio(wheelSpeed, speed);

    // calculate longitudinal force (2 wheel drive)
    longitudinalForce = 2 * calculateTireForcePacejka(slipRatio);
    longitudinalForce = std::max(-30000.0f, std::min(longitudinalForce, 30000.0f));
    
    // Add braking effect
    if (brake > 0.0f) {
        // Simple braking model - apply opposite force to current motion
        float brakeForce = brake * 20000.0f; // Maximum braking force
        if (speed > 0.0f) {
            longitudinalForce -= brakeForce;
        } else if (speed < 0.0f) {
            longitudinalForce += brakeForce;
        }
    }

    if (std::abs(speed) < 1.0f) {
        acceleration.x *= (std::abs(speed) + 0.1f);
    }

    // calculate acceleration
    acceleration.x = longitudinalForce / config.mass;
    acceleration.z = 0.0f; // No lateral acceleration in this context
    acceleration.y = 0.0f; // No vertical acceleration in this context

    // update speed
    speed += acceleration.x * deltaTime;
    if (speed < config.minSpeed) speed = config.minSpeed; // Prevent negative speed
}

float Car::getTotalResistanceForces(float deltaTime) {
    // calculate aerodynamic drag
    dragForce = 0.5f * config.width * config.height * 0.3f * 1.225f * speed * speed;

    // calculate rolling resistance
    rollingResistance = 0.01f * config.mass * 9.81f * (1.0f + speed * 0.01f);
    float totalResistance = dragForce + rollingResistance;
    return totalResistance;
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

float Car::calculateSlipRatio(float wheelSpeed, float carSpeed) const {
    // Convert wheel angular velocity from RPM to rad/s
    float wheelSpeedRadPS = engineSpeed / config.gearRatio * (2.0f * PI / 60.0f);
    // Calculate wheel linear velocity in m/s
    float wheelLinearSpeed = wheelSpeedRadPS * config.tireRadius;
    // Safe slip ratio calculation
    float speedAbs = std::abs(speed);
    const float minSpeed = 0.5f;
    float result = 0.0f;
    if (speedAbs < minSpeed) {
        // Gradual transition when nearly stopped
        result = (wheelLinearSpeed - speed) / minSpeed;
    } else {
        // Normal calculation with proper sign maintenance
        result = (wheelLinearSpeed - speed) / speedAbs;
    }
    // Clamp to reasonable values
    result = std::max(-1.0f, std::min(slipRatio, 1.0f));

    return result;
}

float Car::getEngineTorque(float throttle, float rpm) const {
    return throttle * (400.0f + 250.0f * (rpm / 4000.0f) * (1.0f - rpm / 8000.0f));}

} // namespace CarGame