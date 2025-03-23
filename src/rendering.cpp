#include "../include/rendering.h"

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