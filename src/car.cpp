#include "../include/car.h"

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
    float frictionForce,
    float accelerationForce,
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
      frictionForce(frictionForce),
      accelerationForce(accelerationForce),
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
      config(),
      engineSpeed(0.0f),
      engineSpeed_dot(0.0f),
      throttle(0.0f),
      brake(0.0f) {
}

// Updates the car's speed, rotation, and position based on keyboard input.
void Car::update(float deltaTime) {
    handleLongitudinalMovement(deltaTime); // Handle acceleration and deceleration
    handleLateralMovement(deltaTime); // Handle steering and lateral movement
}

void Car::handleLongitudinalMovement(float deltaTime) {
    // Handle throttle and brake input
    if (IsKeyDown(KEY_UP)) {
        throttle += 3.0f * deltaTime;
        if (throttle > 1.0f) throttle = 1.0f; 
    } else if (IsKeyDown(KEY_DOWN)) {
        brake += 3.0f * deltaTime;
        if (brake > 1.0f) brake = 1.0f;
    }
    else {
        // Gradually reduce throttle and brake when no input
        throttle -= 5.0f * deltaTime;
        brake -= 5.0f * deltaTime;
        if (throttle < 0.0f) throttle = 0.0f;
        if (brake < 0.0f) brake = 0.0f;
    }
}

float Car::getEngineTorque(float throttle, float rpm) const {
    return throttle * (-0.0003f * rpm * rpm + 0.1f * rpm + 500.0f);
}

void Car::applyFriction(float deltaTime) {
    if (speed > 0.0f) {
        speed -= config.getFrictionForce() * deltaTime;
        if (speed < 0.0f) speed = 0.0f;
    } else if (speed < 0.0f) {
        speed += config.getFrictionForce() * deltaTime;
        if (speed > 0.0f) speed = 0.0f;
    }
}

void Car::handleLateralMovement(float deltaTime) {
    float beta = 0.0f;
    
    // Only process steering if car is moving fast enough
    if (std::fabs(speed) > config.getMinMovementSpeed()) {
        processSteeringInput(deltaTime);
        calculateSteering(deltaTime, beta);
    }
    
    updatePosition(deltaTime, beta);
}

void Car::processSteeringInput(float deltaTime) {
    if (IsKeyDown(KEY_LEFT)) {
        steeringSpeed = config.getSteeringSpeed() * deltaTime;
        steeringAngle += steeringSpeed;
        if (steeringAngle > config.getMaxSteeringAngle()) 
            steeringAngle = config.getMaxSteeringAngle();
    }
    else if (IsKeyDown(KEY_RIGHT)) {
        steeringSpeed = -config.getSteeringSpeed() * deltaTime;
        steeringAngle += steeringSpeed;
        if (steeringAngle < -config.getMaxSteeringAngle()) 
            steeringAngle = -config.getMaxSteeringAngle();
    } else {
        // Gradually reduce steering angle when no input
        returnSteeringToCenter(deltaTime);
    }
}

void Car::returnSteeringToCenter(float deltaTime) {
    float returnSpeed = config.getSteeringSpeed() * deltaTime;
    
    if (steeringAngle > 0.0f) {
        steeringAngle -= returnSpeed;
        if (steeringAngle < 0.0f) steeringAngle = 0.0f;
    } else if (steeringAngle < 0.0f) {
        steeringAngle += returnSpeed;
        if (steeringAngle > 0.0f) steeringAngle = 0.0f;
    }
}

void Car::calculateSteering(float deltaTime, float& beta) {
    // Calculate slip angle
    beta = std::atan2f((config.getRearAxleDistance()) * std::tanf(steeringAngle), 
                 (config.getFrontAxleDistance() + config.getRearAxleDistance()));

    // Calculate rotation rate
    float omega_dot = speed * std::cosf(beta) * std::tanf(steeringAngle) / config.getWheelBase();
    
    // Update rotation and normalize to 0-2PI range
    rotation += omega_dot * deltaTime;
    normalizeRotation();
}

void Car::normalizeRotation() {
    if (rotation > 2 * PI) rotation -= 2 * PI;
    if (rotation < 0) rotation += 2 * PI;
}

void Car::updatePosition(float deltaTime, float beta) {
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

} // namespace CarGame