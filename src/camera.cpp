#include "../include/camera.h"
#include <cmath>

namespace CarGame {

GameCamera::GameCamera() {
    // Initialize the camera with default settings
    camera.position = Vector3{ 10.0f, 10.0f, 10.0f };
    camera.target = Vector3{ 0.0f, 0.0f, 0.0f };
    camera.up = Vector3{ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
}

// Updates the camera to follow the car
void GameCamera::update(const Car& car) {
    camera.target = car.getPosition();
    float cameraDistance = 15.0f;
    camera.position.x = car.getPosition().x - cameraDistance * std::sinf(car.getRotation());
    camera.position.z = car.getPosition().z - cameraDistance * std::cosf(car.getRotation());
    camera.position.y = car.getPosition().y + 7.0f;
}

} // namespace CarGame