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

// Define the RAD2DEG macro if it doesn't exist
#ifndef RAD2DEG
    #define RAD2DEG (180.0f/PI)
#endif

const float carWidth = 2.0f;  // Width of the car
const float carLength = 4.0f; // Length of the car
const float wheelBase = 2.8f;
const float l_r = 1.2f;
const float l_f = 1.6f;
const float carHeight = 1.0f; // Height of the car
const float steeringSpeedConstant = 1.0f;

// Function prototypes
void UpdateCar(Car *car, float deltaTime);
void UpdateCamera(Camera *camera, const Car *car);
void DrawScene(const Camera *camera, const Model *carModel, const Car *car, const Vector3 floorPosition);
void HandleHorizontalMovement(Car *car, float deltaTime);
void HandleLateralMovement(Car *car, float deltaTime);

int main(void) {
    // Initialization
    const int screenWidth = 1280;
    const int screenHeight = 720;
    InitWindow(screenWidth, screenHeight, "Car");

    // Define and configure the camera
    Camera camera = { 0 };
    camera.position = Vector3{ 10.0f, 10.0f, 10.0f };
    camera.target   = Vector3{ 0.0f, 0.0f, 0.0f };
    camera.up       = Vector3{ 0.0f, 1.0f, 0.0f };
    camera.fovy     = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    // Create a car instance
    Car car = {
        { 0.0f, 0.5f, 0.0f },  // position
        0.0f,                  // speed
        0.0f                   // rotation (radians)
    };

    // Create the car model (a simple box)
    Model carModel = LoadModelFromMesh(GenMeshCube(carWidth, carHeight, carLength));

    // Floor position (a large plane)
    Vector3 floorPosition = { 0.0f, 0.0f, 0.0f };

    SetTargetFPS(60);

    // Main game loop
    while (!WindowShouldClose()) {
        float deltaTime = GetFrameTime();

        // Update game objects
        UpdateCar(&car, deltaTime);
        UpdateCamera(&camera, &car);

        // Draw the scene
        BeginDrawing();
        ClearBackground(RAYWHITE);
        DrawScene(&camera, &carModel, &car, floorPosition);
        EndDrawing();
    }

    // De-Initialization
    UnloadModel(carModel);
    CloseWindow();

    return 0;
}

// Updates the car's speed, rotation, and position based on keyboard input.
void UpdateCar(Car *car, float deltaTime) {
    
    HandleHorizontalMovement(car, deltaTime); // Handle acceleration and deceleration
    HandleLateralMovement(car, deltaTime); // Handle steering and lateral movement
}

void HandleHorizontalMovement(Car *car, float deltaTime) {
    // Accelerate or decelerate the car
    if (IsKeyDown(KEY_UP)) {
        car->speed += 5.0f * deltaTime;
        if (car->speed > 10.0f) car->speed = 10.0f;
    } else if (IsKeyDown(KEY_DOWN)) {
        car->speed -= 5.0f * deltaTime;
        if (car->speed < -5.0f) car->speed = -5.0f;
    } else {
        // Apply friction to gradually reduce speed
        if (car->speed > 0.0f) {
            car->speed -= 2.0f * deltaTime;
            if (car->speed < 0.0f) car->speed = 0.0f;
        } else if (car->speed < 0.0f) {
            car->speed += 2.0f * deltaTime;
            if (car->speed > 0.0f) car->speed = 0.0f;
        }
    }
}

void HandleLateralMovement(Car *car, float deltaTime) {
    // Steering: only allow steering when the car is moving
    if (fabs(car->speed) > 0.1f) {
        if (IsKeyDown(KEY_LEFT)) {
            car->steeringSpeed = steeringSpeedConstant * deltaTime;
            car->steeringAngle += car->steeringSpeed;
            if (car->steeringAngle > 0.5f) car->steeringAngle = 0.5f;
        }
        else  if (IsKeyDown(KEY_RIGHT)) {
            car->steeringSpeed = -steeringSpeedConstant * deltaTime;
            car->steeringAngle += car->steeringSpeed;
            if (car->steeringAngle < -0.5f) car->steeringAngle = -0.5f;
        } else {
            // Gradually reduce steering angle when no input
            if (car->steeringAngle > 0.0f) {
                car->steeringAngle -= 1.0f * deltaTime;
                if (car->steeringAngle < 0.0f) car->steeringAngle = 0.0f;
            } else if (car->steeringAngle < 0.0f) {
                car->steeringAngle += 1.0f * deltaTime;
                if (car->steeringAngle > 0.0f) car->steeringAngle = 0.0f;
            }
        }     
    }

    float beta = atan2f((l_r) * tanf(car->steeringAngle), (l_f + l_r));

    float omega_dot = car->speed * cosf(beta) * tanf(car->steeringAngle) / wheelBase;
    car->rotation += omega_dot * deltaTime; // Update rotation based on steering angle
    if (car->rotation > 2 * PI) car->rotation -= 2 * PI;
    if (car->rotation < 0) car->rotation += 2 * PI; 

    // Update car's velocity based on speed and rotation
    car->velocity.z = car->speed * cosf(car->rotation + beta);
    car->velocity.x = car->speed * sinf(car->rotation + beta);

    // Update car's position based on velocity
    car->position.x += car->velocity.x * deltaTime;
    car->position.z += car->velocity.z * deltaTime;

    // Keep the car above the ground
    if (car->position.y < 0.5f) {
        car->position.y = 0.5f;
    }
}

// Updates the camera to follow the car.
void UpdateCamera(Camera *camera, const Car *car) {
    camera->target = car->position;
    float cameraDistance = 15.0f;
    camera->position.x = car->position.x - cameraDistance * sinf(car->rotation);
    camera->position.z = car->position.z - cameraDistance * cosf(car->rotation);
    camera->position.y = car->position.y + 7.0f;
}

// Draws the 3D scene, including the floor and the car, and overlays UI text.
void DrawScene(const Camera *camera, const Model *carModel, const Car *car, const Vector3 floorPosition) {
    BeginMode3D(*camera);
        // Draw the ground plane
        DrawPlane(floorPosition, Vector2{50.0f, 50.0f}, LIGHTGRAY);
        // Draw the car model with the correct rotation (converted from radians to degrees)
        DrawModelEx(*carModel, car->position, Vector3{0.0f, 1.0f, 0.0f}, car->rotation * RAD2DEG, Vector3{1.0f, 1.0f, 1.0f}, MAROON);
    EndMode3D();

    // Draw overlay text instructions and car status
    DrawText(TextFormat("Car Speed: %.2f", car->speed), 10, 10, 20, BLACK);
    DrawText(TextFormat("Car Rotation: %.2f degrees", car->rotation * RAD2DEG), 10, 40, 20, BLACK);
    DrawText(TextFormat("Steering Angle: %.2f degrees", car->steeringAngle * RAD2DEG), 10, 70, 20, BLACK);
    DrawText(TextFormat("Car Position: (%.2f, %.2f, %.2f)", car->position.x, car->position.y, car->position.z), 10, 100, 20, BLACK);
}