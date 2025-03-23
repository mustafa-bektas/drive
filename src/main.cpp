#include "../include/car.h"
#include "../include/camera.h"
#include "../include/rendering.h"

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

    // Create a car instance using the new creation function
    Vector3 startPosition = { 0.0f, 0.5f, 0.0f };
    Car car = CreateCar(startPosition);

    // Create the car model (a simple box) using the car's config properties
    Model carModel = LoadModelFromMesh(GenMeshCube(
        car.config.width, 
        car.config.height, 
        car.config.length
    ));

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