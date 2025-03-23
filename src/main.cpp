#include "../include/car.h"
#include "../include/camera.h"
#include "../include/rendering.h"
#include "../include/input_handler.h"
#include <memory>

using namespace CarGame;

int main(void) {
    // Initialization
    const int screenWidth = 1280;
    const int screenHeight = 720;
    InitWindow(screenWidth, screenHeight, "Car Game");

    Vector3 startPosition = { 0.0f, 0.5f, 0.0f };
    Car car(startPosition);
    GameCamera camera;
    Renderer renderer;
    InputHandler inputHandler;
    
    renderer.initialize(car);

    Vector3 floorPosition = { 0.0f, 0.0f, 0.0f };

    SetTargetFPS(60);

    // Main game loop
    while (!WindowShouldClose()) {
        float deltaTime = GetFrameTime();

        // 1. Process Input
        inputHandler.processInput(car, deltaTime);
        
        // 2. Update Game Logic
        car.update(deltaTime);
        camera.update(car);

        // 3. Render
        BeginDrawing();
        ClearBackground(RAYWHITE);
        renderer.drawScene(camera, car, floorPosition);
        EndDrawing();
    }

    CloseWindow();

    return 0;
}