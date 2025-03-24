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
    float mass,
    float corneringStiffnessFront,
    float corneringStiffnessRear,
    float pacejkaB_lat,
    float pacejkaC_lat,
    float pacejkaD_lat,
    float pacejkaE_lat,
    float frontWeight,
    float normalLoadFront,
    float normalLoadRear,
    float heightCG
)
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
      mass(mass),
      corneringStiffnessFront(corneringStiffnessFront),
      corneringStiffnessRear(corneringStiffnessRear),
      pacejkaB_lat(pacejkaB_lat),
      pacejkaC_lat(pacejkaC_lat),
      pacejkaD_lat(pacejkaD_lat),
      pacejkaE_lat(pacejkaE_lat),
      frontWeight(frontWeight),
      normalLoadFront(normalLoadFront),
      normalLoadRear(normalLoadRear),
      heightCG(heightCG) {
}

Car::Car(const Vector3& startPosition) 
    : position(startPosition),
      velocity({0.0f, 0.0f, 0.0f}),
      acceleration({0.0f, 0.0f, 0.0f}),
      speed(0.0f),
      rotation(0.0f),
      steeringAngle(0.0f),
      steeringSpeed(0.0f),
      config(),
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
      netForce(0.0f),
      slipAngleFront(0.0f),
      slipAngleRear(0.0f),
      lateralForceFront(0.0f),
      lateralForceRear(0.0f),
      yawMoment(0.0f),
      yawRate(0.0f),
      lateralVelocity(0.0f),
      sideSlipAngle(0.0f),
      sideSlipAngleDot(0.0f),
      yawAngle(0.0f),
      yawAngleDot(0.0f),
      lateralAcceleration(0.0f),
      yawAngleDotDot(0.0f) {
}

void Car::update(float deltaTime) {
    updateLongitudinalPhysics(deltaTime);
    updateLateralPhysics(deltaTime);
}

void Car::updateLongitudinalPhysics(float deltaTime) {
    const float idleRPM = 1.0f;
    
    if (throttle > 0.1f) {
        clutch = false;
        engineSpeed_dot = getEngineTorque(throttle, engineSpeed) / config.inertiaAtEngine;
        engineSpeed += engineSpeed_dot * deltaTime;
        if (engineSpeed > 8000.0f) engineSpeed = 8000.0f;
    } else if (std::abs(speed) < 0.5f) {
        engineSpeed = idleRPM;
        clutch = true;
    } else {
        float wheelRPM = std::abs(speed) / config.tireRadius * config.gearRatio * (60.0f / (2.0f * PI));
        engineSpeed = std::max(idleRPM, wheelRPM);
    }
    
    if (!clutch)
    {
        if (engineSpeed > 8000.0f) engineSpeed = 8000.0f;
        
        float engineTorque = getEngineTorque(throttle, engineSpeed);
        
        if (throttle < 0.1f && engineSpeed > 1000.0f) {
            engineTorque -= (engineSpeed / 8000.0f) * 75.0f;
        }

        wheelRotationSpeed = clutch ? wheelRotationSpeed : engineSpeed / config.gearRatio * (2.0f * PI / 60.0f);

        float brakeTorque = 0.0f;
        if (brake > 0.0f) {
            brakeTorque = brake * 8000.0f; 
            
            if (wheelRotationSpeed > 0.0f) {
                brakeTorque = -brakeTorque;
            } else if (wheelRotationSpeed < 0.0f) {
            } else {
                brakeTorque = 0.0f;
            }
        }

        float wheelInertia = 5.0f; 
        float wheelAngularAccel = (engineTorque + brakeTorque) / wheelInertia;
        
        wheelRotationSpeed += wheelAngularAccel * deltaTime;
        float wheelLinearSpeed = wheelRotationSpeed * config.tireRadius;

        slipRatio = calculateSlipRatio(wheelLinearSpeed, speed);    
        longitudinalForce = 2 * calculateTireForcePacejka(slipRatio);
                
        float totalResistance = getTotalResistanceForces(deltaTime);
        netForce = longitudinalForce - totalResistance;

        acceleration.x = netForce / config.mass;
        speed += acceleration.x * deltaTime;
    }
}

float Car::getTotalResistanceForces(float deltaTime) {
    const float airDensity = 1.225f;
    const float dragCoefficient = 0.3f;
    float frontalArea = config.width * config.height * 0.8f;

    dragForce = 0.5f * airDensity * dragCoefficient * frontalArea * speed * speed;

    float rollingCoefficient = 0.015f * (1.0f + std::abs(speed) * 0.01f);
    rollingResistance = rollingCoefficient * config.mass * 9.81f;
    
    if (speed != 0.0f) {
        rollingResistance *= (speed > 0.0f ? 1.0f : -1.0f);
    }
    
    return dragForce + rollingResistance;
}

float Car::calculateTireForcePacejka(float slipRatio) const {
    float D = 1.0f;
    float C = 1.5f;
    float B = 10.0f;
    float E = 0.1f;
    float Fz = 4000.0f;

    float coefficient = D * std::sinf(C * std::atanf(B * slipRatio - E * (B * slipRatio - std::atanf(B * slipRatio))));
    
    return coefficient * Fz;
}

float Car::calculateSlipRatio(float wheelLinearSpeed, float vehicleSpeed) const {
    float speedAbs = std::abs(vehicleSpeed);
    const float minSpeed = 0.5f;
    float result;
    
    if (speedAbs < minSpeed) {
        result = (wheelLinearSpeed - vehicleSpeed) / minSpeed;
    } else {
        result = (wheelLinearSpeed - vehicleSpeed) / speedAbs;
    }
    
    result = std::max(-1.0f, std::min(result, 1.0f));
    return result;
}

float Car::getEngineTorque(float throttle, float rpm) const {
    return throttle * (400.0f + 250.0f * (rpm / 4000.0f) * (1.0f - rpm / 8000.0f));
}

void Car::normalizeRotation() {
    if (rotation > 2 * PI) rotation -= 2 * PI;
    if (rotation < 0) rotation += 2 * PI;
}

float Car::calculatePacejkaLateral(float slipAngle, float Fz, bool isFrontTire) const {
    float B = config.pacejkaB_lat;
    float C = config.pacejkaC_lat;
    float D = config.pacejkaD_lat;
    float E = config.pacejkaE_lat;
    
    float argument = B * slipAngle - E * (B * slipAngle - std::atan(B * slipAngle));
    float peak = D * Fz;
    
    float lateralForce = peak * std::sin(C * std::atan(argument));
    
    return -lateralForce;
}

void Car::updateLateralPhysics(float deltaTime) {
    float yawInertia = config.mass * (std::pow(config.wheelBase, 2) + std::pow(config.width, 2)) / 12.0f;
    const float MIN_SPEED = 2.0f;
    
    if (std::fabs(speed) < MIN_SPEED) {
        slipAngleFront *= 0.8f;
        slipAngleRear *= 0.8f;
        lateralVelocity *= 0.8f;
        yawRate *= 0.8f;
        
        if (std::fabs(steeringAngle) > 0.01f) {
            float turnRate = steeringAngle * std::fabs(speed) / config.wheelBase;
            yawRate = speed >= 0 ? turnRate : -turnRate;
        }
        
        rotation += yawRate * deltaTime;
        normalizeRotation();
        
        velocity.x = speed * std::sinf(rotation);
        velocity.z = speed * std::cosf(rotation);
        
        position.x += velocity.x * deltaTime;
        position.z += velocity.z * deltaTime;
        return;
    }
        
    float vx = speed;
    
    float vFront = lateralVelocity + config.frontAxleDistance * yawRate;
    float vRear = lateralVelocity - config.rearAxleDistance * yawRate;

    slipAngleFront = std::atan2f(vFront, std::fabs(vx)) - (vx >= 0 ? steeringAngle : -steeringAngle);
    slipAngleRear = std::atan2f(vRear, std::fabs(vx));
    
    float normalLoadFront = config.normalLoadFront;
    float normalLoadRear = config.normalLoadRear;
    
    float loadTransferLong = config.mass * acceleration.x * config.heightCG / config.wheelBase;
    normalLoadFront -= loadTransferLong;
    normalLoadRear += loadTransferLong;
    
    const float MIN_LOAD = 500.0f;
    normalLoadFront = std::max(MIN_LOAD, normalLoadFront);
    normalLoadRear = std::max(MIN_LOAD, normalLoadRear);
    
    lateralForceFront = 1 * calculatePacejkaLateral(slipAngleFront, normalLoadFront, true);
    lateralForceRear = 1 * calculatePacejkaLateral(slipAngleRear, normalLoadRear, false);
    
    float totalLateralForce = lateralForceFront + lateralForceRear;
    yawMoment = lateralForceFront * config.frontAxleDistance - lateralForceRear * config.rearAxleDistance;
    
    lateralAcceleration = totalLateralForce / config.mass;
    float yawAcceleration = yawMoment / yawInertia;
    
    const float YAW_DAMPING = 0.45f;
    const float LATERAL_DAMPING = 0.45f;
    
    yawAcceleration -= yawRate * YAW_DAMPING;
    lateralAcceleration -= lateralVelocity * LATERAL_DAMPING;
    
    const float MAX_LATERAL_ACCEL = 40.0f;
    const float MAX_YAW_ACCEL = 10.0f;
    
    lateralAcceleration = std::max(-MAX_LATERAL_ACCEL, std::min(lateralAcceleration, MAX_LATERAL_ACCEL));
    yawAcceleration = std::max(-MAX_YAW_ACCEL, std::min(yawAcceleration, MAX_YAW_ACCEL));
    
    lateralVelocity += lateralAcceleration * deltaTime;
    yawRate += yawAcceleration * deltaTime;
    
    const float MAX_LATERAL_VEL = 20.0f;
    const float MAX_YAW_RATE = 2.0f;
    
    lateralVelocity = std::max(-MAX_LATERAL_VEL, std::min(lateralVelocity, MAX_LATERAL_VEL));
    yawRate = std::max(-MAX_YAW_RATE, std::min(yawRate, MAX_YAW_RATE));
    
    rotation += yawRate * deltaTime;
    normalizeRotation();
    
    velocity.x = vx * std::sinf(rotation) + lateralVelocity * std::cosf(rotation);
    velocity.z = vx * std::cosf(rotation) - lateralVelocity * std::sinf(rotation);
    
    position.x += velocity.x * deltaTime;
    position.z += velocity.z * deltaTime;
    
    if (position.y < 0.5f) {
        position.y = 0.5f;
    }
}
} // namespace CarGame