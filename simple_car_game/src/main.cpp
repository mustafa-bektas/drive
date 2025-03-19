#include "raylib.h"
#include <cmath>
#include <vector>

// Constants
const int screenWidth = 800;
const int screenHeight = 600;
const float roadWidth = 10.0f;
const float laneWidth = 3.5f;
const float roadLength = 100.0f;
const float laneMarkingLength = 3.0f;
const float laneMarkingGap = 5.0f;

// Car properties
struct Car {
    Vector3 position;
    Vector3 size;
    float speed;
    float rotation;
    Color color;
};

// Lane marking properties
struct LaneMarking {
    Vector3 position;
    Vector3 size;
    Color color;
};

// Generate lane markings
std::vector<LaneMarking> generateLaneMarkings() {
    std::vector<LaneMarking> markings;
    
    // Calculate how many markings we need along the road
    int numMarkings = static_cast<int>(roadLength / (laneMarkingLength + laneMarkingGap));
    
    // Create lane markings for the center of the road
    for (int i = 0; i < numMarkings; i++) {
        float zPos = -i * (laneMarkingLength + laneMarkingGap);
        
        LaneMarking marking;
        marking.position = {0.0f, 0.01f, zPos - laneMarkingLength/2};  // Slightly above the road to avoid z-fighting
        marking.size = {0.15f, 0.01f, laneMarkingLength};
        marking.color = WHITE;
        
        markings.push_back(marking);
    }
    
    return markings;
}

int main() {
    // Initialize the window
    InitWindow(screenWidth, screenHeight, "Simple Car Game");
    
    // Define the camera
    Camera camera = { 0 };
    camera.position = {0.0f, 2.5f, 10.0f};  // Camera position
    camera.target = {0.0f, 0.0f, 0.0f};     // Camera target (looking at)
    camera.up = {0.0f, 1.0f, 0.0f};         // Camera up vector
    camera.fovy = 45.0f;                    // Camera field-of-view Y
    camera.projection = CAMERA_PERSPECTIVE;  // Camera projection type
    
    // Define the car
    Car car;
    car.position = {0.0f, 0.5f, 0.0f};  // Position (x, y, z)
    car.size = {1.0f, 0.5f, 2.0f};      // Size (width, height, length)
    car.speed = 0.0f;
    car.rotation = 0.0f;
    car.color = RED;
    
    // Generate lane markings
    std::vector<LaneMarking> laneMarkings = generateLaneMarkings();
    
    // Set target FPS
    SetTargetFPS(60);
    
    // Main game loop
    while (!WindowShouldClose()) {
        // Update
        
        // Car controls
        if (IsKeyDown(KEY_UP)) {
            car.speed += 0.01f;
        } else if (IsKeyDown(KEY_DOWN)) {
            car.speed -= 0.01f;
        } else {
            // Apply friction to slow down the car
            if (car.speed > 0.0f) {
                car.speed -= 0.005f;
            } else if (car.speed < 0.0f) {
                car.speed += 0.005f;
            }
            
            // Prevent very small values
            if (fabs(car.speed) < 0.005f) {
                car.speed = 0.0f;
            }
        }
        
        // Limit max speed
        if (car.speed > 0.3f) car.speed = 0.3f;
        if (car.speed < -0.15f) car.speed = -0.15f;
        
        // Steering
        if (IsKeyDown(KEY_RIGHT)) {
            car.rotation -= 2.0f * car.speed;
        } else if (IsKeyDown(KEY_LEFT)) {
            car.rotation += 2.0f * car.speed;
        }
        
        // Update car position based on speed and rotation
        float radians = car.rotation * DEG2RAD;
        car.position.x += sin(radians) * car.speed;
        car.position.z -= cos(radians) * car.speed;
        
        // Keep the car within the road bounds
        if (car.position.x > roadWidth/2 - car.size.x/2) {
            car.position.x = roadWidth/2 - car.size.x/2;
        }
        if (car.position.x < -roadWidth/2 + car.size.x/2) {
            car.position.x = -roadWidth/2 + car.size.x/2;
        }
        
        // Update camera position to follow the car
        camera.position = {
            car.position.x,
            car.position.y + 3.5f,
            car.position.z + 7.0f
        };
        camera.target = car.position;
        
        // Drawing
        BeginDrawing();
            ClearBackground(SKYBLUE);
            
            BeginMode3D(camera);
                // Draw road
                DrawPlane({0.0f, 0.0f, -roadLength/2}, {roadWidth, roadLength}, DARKGRAY);
                
                // Draw lane markings
                for (const auto& marking : laneMarkings) {
                    DrawCube(
                        {marking.position.x + car.position.x, marking.position.y, marking.position.z},
                        marking.size.x, marking.size.y, marking.size.z,
                        marking.color
                    );
                }
                
                // Draw car (as a box)
                DrawCube(car.position, car.size.x, car.size.y, car.size.z, car.color);
                
            EndMode3D();
            
            // UI elements
            DrawText("Use arrow keys to drive", 10, 10, 20, WHITE);
            DrawText(TextFormat("Speed: %.2f", car.speed), 10, 40, 20, WHITE);
            
        EndDrawing();
    }
    
    // Clean up
    CloseWindow();
    
    return 0;
}
