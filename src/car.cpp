#include "../include/car.h"
#include "car.h"

namespace CarGame {

CarPhysicsConfig::CarPhysicsConfig(
    float width,
    float length,
    float height,
    float wheelBase,
    float frontAxleDistance,
    float rearAxleDistance,
    float maxSpeed,
    float minSpeed,
    float maxSteeringAngle,
    float steeringSpeed,
    float minMovementSpeed,
    int gearRatio,
    float tireRadius,
    float inertiaAtEngine)
    : width(width), 
      length(length), 
      height(height),
      wheelBase(wheelBase),
      frontAxleDistance(frontAxleDistance),
      rearAxleDistance(rearAxleDistance),
      maxSpeed(maxSpeed),
      minSpeed(minSpeed),
      maxSteeringAngle(maxSteeringAngle),
      steeringSpeed(steeringSpeed),
      minMovementSpeed(minMovementSpeed),
      gearRatio(gearRatio),
      tireRadius(tireRadius),
      inertiaAtEngine(inertiaAtEngine) {
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
    float engineTorque = getEngineTorque(throttle, engineSpeed);
    float totalLoadTorque = config.gearRatio * config.tireRadius * getTotalResistanceForces(deltaTime);
    engineSpeed_dot = (engineTorque - totalLoadTorque) / config.inertiaAtEngine;
    engineSpeed += engineSpeed_dot * deltaTime;
    if (engineSpeed < 0.0f) engineSpeed = 0.0f; // Prevent negative engine speed
    if (engineSpeed > 7000.0f) engineSpeed = 7000.0f; // Prevent excessive engine speed
    float wheelSpeed = engineSpeed / config.gearRatio;

    // calculate longitudinal slip ratio
    slipRatio = (wheelSpeed * config.tireRadius - speed) / (speed + 0.01f);

    // calculate longitudinal force
    longitudinalForce = calculateTireForcePacejka(slipRatio);

    // calculate acceleration
    acceleration.x = longitudinalForce / config.inertiaAtEngine;
    acceleration.z = 0.0f; // No lateral acceleration in this context
    acceleration.y = 0.0f; // No vertical acceleration in this context

    // update speed
    speed += acceleration.x * deltaTime;
    if (speed < config.minSpeed) speed = config.minSpeed; // Prevent negative speed
}

float Car::getTotalResistanceForces(float deltaTime) {
    // calculate aerodynamic drag
    dragForce = 0.5f * config.width * config.height * 0.3f * speed * speed;

    // calculate rolling resistance
    rollingResistance = 0.01f * 1000 * 9.81f; // Assuming a constant coefficient of rolling resistance
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

float Car::calculateTireForcePacejka(float slipRatio) const
{
    float D = 1.0f; // Peak force
    float C = 1.0f; // Stiffness factor
    float B = 1.0f; // Shape factor
    float E = 0.1f; // Curvature factor
    float Fz = 1.0f; // Normal load

    // Pacejka tire model
    float force = D * std::sinf(C * std::atanf(B * slipRatio - E * (B * slipRatio - std::atanf(B * slipRatio)))) * Fz;
    return force;
}

float Car::getEngineTorque(float throttle, float rpm) const {
    return throttle * (-0.0003f * rpm * rpm + 0.1f * rpm + 500.0f);
}

} // namespace CarGame