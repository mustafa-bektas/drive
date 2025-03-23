#include "../include/camera.h"
#include <math.h>

// Updates the camera to follow the car.
void UpdateCamera(Camera *camera, const Car *car) {
    camera->target = car->position;
    float cameraDistance = 15.0f;
    camera->position.x = car->position.x - cameraDistance * sinf(car->rotation);
    camera->position.z = car->position.z - cameraDistance * cosf(car->rotation);
    camera->position.y = car->position.y + 7.0f;
}