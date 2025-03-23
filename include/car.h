#ifndef CAR_H
#define CAR_H

#include "raylib.h"
#include <math.h>

// Car structure
typedef struct {
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
    float speed;
    float rotation;  // in radians
    float steeringAngle; // in radians
    float steeringSpeed; // in radians per second
} Car;

// Car constants
extern const float carWidth;
extern const float carLength;
extern const float wheelBase;
extern const float l_r;
extern const float l_f;
extern const float carHeight;
extern const float steeringSpeedConstant;

// Define the RAD2DEG macro if it doesn't exist
#ifndef RAD2DEG
    #define RAD2DEG (180.0f/PI)
#endif

// Function prototypes
void UpdateCar(Car *car, float deltaTime);
void HandleHorizontalMovement(Car *car, float deltaTime);
void HandleLateralMovement(Car *car, float deltaTime);

#endif // CAR_H