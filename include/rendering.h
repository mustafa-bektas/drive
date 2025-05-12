#pragma once

#include "raylib.h"
#include "car.h"
#include "camera.h"
#include <vector>
#include <functional>

namespace CarGame {


class Renderer {
public:
    Renderer();
    ~Renderer();
    
    void initialize(const Car& car);
    
    void drawScene(const GameCamera& camera, const Car& car, const Vector3& floorPosition);
    
private:
    void draw3DScene(const GameCamera& camera, const Car& car, const Vector3& floorPosition);
    
    void drawTelemetryPanel(const Car& car);
    void drawInstructions();
    
    int drawSection(int x, int y, const char* title, 
                   const std::vector<std::function<void(int, int)>>& drawFuncs);
    
    Model carModel;

    // Cached resources for road rendering
    Mesh roadSegmentMesh;
    Mesh dividerSegmentMesh;
    Mesh edgeSegmentMesh;
    Material roadMaterial;
    Material dividerMaterial;
    Material edgeMaterial;
};

} // namespace CarGame
