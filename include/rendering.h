#ifndef RENDERING_H
#define RENDERING_H

#include "raylib.h"
#include "car.h"

// Function prototypes
void DrawScene(const Camera *camera, const Model *carModel, const Car *car, const Vector3 floorPosition);

#endif // RENDERING_H