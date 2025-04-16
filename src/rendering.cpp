#include "../include/rendering.h"
#include "../include/road_geometry.h"
#include "raylib.h"
#include "raymath.h"
#include <string>
#include <array>
#include <vector>
#include <functional>
#include "rlgl.h"

namespace CarGame {

// ui stuff
namespace UI {
    constexpr int FontSize = 20;
    constexpr int SmallFontSize = 16;
    constexpr int LineHeight = 24;
    constexpr int PanelMargin = 10;
    constexpr int SectionSpacing = 5;
    constexpr int PanelWidth = 310;
    constexpr int PanelHeight = 700;
    
    inline int GetRightPanelX() {
        return GetScreenWidth() - 400;
    }
    
    const Color TextColor = BLACK;
    const Color HeaderColor = DARKBLUE;
    const Color SectionColor = DARKGRAY;
    const Color PanelColor = { 200, 200, 200, 180 };
}

Renderer::Renderer() = default;

Renderer::~Renderer() {
    UnloadModel(carModel);
}

void Renderer::initialize(const Car& car) {
    // make car model from cube
    carModel = LoadModelFromMesh(GenMeshCube(
        car.config.width, 
        car.config.height, 
        car.config.length
    ));
}

void Renderer::drawScene(const GameCamera& camera, const Car& car, const Vector3& floorPosition) {
    draw3DScene(camera, car, floorPosition);
    drawTelemetryPanel(car);
    drawInstructions();
}

void Renderer::draw3DScene(const GameCamera& camera, const Car& car, const Vector3& floorPosition) {
    // nice sky background
    DrawRectangleGradientV(0, 0, GetScreenWidth(), GetScreenHeight(), 
                          SKYBLUE, DARKBLUE);
    
    BeginMode3D(camera.getCamera());
        const float groundSize = 500.0f;
        const int gridSpacing = 10;
        
        // ground plane
        DrawPlane(floorPosition, { groundSize, groundSize }, GREEN);

        // draw curved road
        const float laneWidth = 7.5f; // narrower lanes 
        const float roadWidth = 3 * laneWidth; // three lanes total
        const float segmentLength = 2.0f; // segment size for drawing
        const float roadThickness = 0.02f; // road height from ground

        for (float z = -groundSize / 2.0f; z < groundSize / 2.0f; z += segmentLength) {
            float currentCenterZ = z + segmentLength / 2.0f; // middle z of segment
            float nextCenterZ = z + segmentLength + segmentLength / 2.0f;

            // get road curve position and angle
            float currentCenterX = RoadGeometry::getRoadCenterlineX(currentCenterZ);
            float currentAngle = RoadGeometry::getRoadTangentAngle(currentCenterZ); // in radians

            // position and rotation
            Vector3 segmentCenter = { currentCenterX, roadThickness / 2.0f, currentCenterZ };
            Vector3 rotationAxis = { 0.0f, 1.0f, 0.0f }; // y-axis rotation
            float rotationAngleDegrees = currentAngle * RAD2DEG;

            // create meshes for road parts
            Mesh roadSegmentMesh = GenMeshCube(roadWidth, roadThickness, segmentLength);
            Mesh dividerSegmentMesh = GenMeshCube(0.5f, roadThickness, segmentLength);
            Mesh edgeSegmentMesh = GenMeshCube(0.3f, roadThickness, segmentLength);

            // transform for this segment
            Matrix segmentTransform = MatrixMultiply(MatrixRotate(rotationAxis, currentAngle), MatrixTranslate(segmentCenter.x, segmentCenter.y, segmentCenter.z));

            // set up materials and colors
            Material roadMaterial = LoadMaterialDefault();
            roadMaterial.maps[MATERIAL_MAP_DIFFUSE].color = DARKGRAY;
            Material dividerMaterial = LoadMaterialDefault();
            dividerMaterial.maps[MATERIAL_MAP_DIFFUSE].color = WHITE;
            Material edgeMaterial = LoadMaterialDefault();
            edgeMaterial.maps[MATERIAL_MAP_DIFFUSE].color = RED;

            // draw main road surface
            DrawMesh(roadSegmentMesh, roadMaterial, segmentTransform);

            // calculate direction vectors
            Vector2 tangent = RoadGeometry::getRoadTangentVector(currentCenterZ);
            Vector2 normal = RoadGeometry::getRoadNormalVector(currentCenterZ); // perpendicular to road

            // left lane divider
            Vector3 leftDividerOffset = { normal.x * (laneWidth / 2.0f), 0.0f, normal.y * (laneWidth / 2.0f) };
            Vector3 leftDividerPos = Vector3Add(segmentCenter, leftDividerOffset);
            leftDividerPos.y += roadThickness / 2.0f; // slightly above road

            // right lane divider
            Vector3 rightDividerOffset = { normal.x * (-laneWidth / 2.0f), 0.0f, normal.y * (-laneWidth / 2.0f) };
            Vector3 rightDividerPos = Vector3Add(segmentCenter, rightDividerOffset);
            rightDividerPos.y += roadThickness / 2.0f; // slightly above road

            // make dashed lines
            if (static_cast<int>(z / segmentLength) % 4 < 2) { // skip every other set
                 Matrix leftDividerTransform = MatrixMultiply(MatrixRotate(rotationAxis, currentAngle), MatrixTranslate(leftDividerPos.x, leftDividerPos.y, leftDividerPos.z));
                 Matrix rightDividerTransform = MatrixMultiply(MatrixRotate(rotationAxis, currentAngle), MatrixTranslate(rightDividerPos.x, rightDividerPos.y, rightDividerPos.z));
                 DrawMesh(dividerSegmentMesh, dividerMaterial, leftDividerTransform); 
                 DrawMesh(dividerSegmentMesh, dividerMaterial, rightDividerTransform);
            }

            // left edge position
            Vector3 leftEdgeOffset = { normal.x * (roadWidth / 2.0f), 0.0f, normal.y * (roadWidth / 2.0f) };
            Vector3 leftEdgePos = Vector3Add(segmentCenter, leftEdgeOffset);
            leftEdgePos.y += roadThickness / 2.0f;

            // right edge position
            Vector3 rightEdgeOffset = { normal.x * (-roadWidth / 2.0f), 0.0f, normal.y * (-roadWidth / 2.0f) };
            Vector3 rightEdgePos = Vector3Add(segmentCenter, rightEdgeOffset);
            rightEdgePos.y += roadThickness / 2.0f;

            // draw road edges
            Matrix leftEdgeTransform = MatrixMultiply(MatrixRotate(rotationAxis, currentAngle), MatrixTranslate(leftEdgePos.x, leftEdgePos.y, leftEdgePos.z));
            Matrix rightEdgeTransform = MatrixMultiply(MatrixRotate(rotationAxis, currentAngle), MatrixTranslate(rightEdgePos.x, rightEdgePos.y, rightEdgePos.z));
            DrawMesh(edgeSegmentMesh, edgeMaterial, leftEdgeTransform); 
            DrawMesh(edgeSegmentMesh, edgeMaterial, rightEdgeTransform);

            // clean up resources
            UnloadMesh(roadSegmentMesh);
            UnloadMesh(dividerSegmentMesh);
            UnloadMesh(edgeSegmentMesh);
            UnloadMaterial(roadMaterial);
            UnloadMaterial(dividerMaterial);
            UnloadMaterial(edgeMaterial);
        }
        // end of road drawing

        // grid for reference (optional, can be kept or removed)
        for (int i = -gridSpacing; i <= gridSpacing; i++) {
            DrawLine3D(
                {-groundSize/2, 0.01f, i * (groundSize/gridSpacing/2)},
                {groundSize/2, 0.01f, i * (groundSize/gridSpacing/2)},
                {0, 100, 0, 80}
            );
            
            DrawLine3D(
                {i * (groundSize/gridSpacing/2), 0.01f, -groundSize/2},
                {i * (groundSize/gridSpacing/2), 0.01f, groundSize/2},
                {0, 100, 0, 80}
            );
        }
        
        // trees on road sides (adjust position based on curved road)
        const float treeOffsetDistance = 20.0f; // Distance from the road centerline
        for (float z = -groundSize / 2.0f; z <= groundSize / 2.0f; z += 20.0f) {
            float roadCenterX = RoadGeometry::getRoadCenterlineX(z);
            Vector2 roadNormal = RoadGeometry::getRoadNormalVector(z);

            // Left tree position
            Vector3 leftTreePos = {
                roadCenterX + roadNormal.x * treeOffsetDistance,
                0.0f, // Base of the tree on the ground
                z + roadNormal.y * treeOffsetDistance
            };
            DrawCylinder(leftTreePos, 0.5f, 0.5f, 5.0f, 8, BROWN);
            DrawSphere(Vector3Add(leftTreePos, {0, 5.0f, 0}), 3.0f, DARKGREEN);

            // Right tree position
            Vector3 rightTreePos = {
                roadCenterX - roadNormal.x * treeOffsetDistance,
                0.0f,
                z - roadNormal.y * treeOffsetDistance
            };
            DrawCylinder(rightTreePos, 0.5f, 0.5f, 5.0f, 8, BROWN);
            DrawSphere(Vector3Add(rightTreePos, {0, 5.0f, 0}), 3.0f, DARKGREEN);
        }

        // draw car
        DrawModelEx(
            carModel, 
            car.position, 
            { 0.0f, 1.0f, 0.0f },
            car.rotation * RAD2DEG,
            { 1.0f, 1.0f, 1.0f },
            MAROON
        );
    EndMode3D();
}

void Renderer::drawTelemetryPanel(const Car& car) {
    int textX = UI::GetRightPanelX();
    int textY = 20;
    
    // panel bg
    DrawRectangle(
        textX - UI::PanelMargin, 
        textY - UI::PanelMargin, 
        UI::PanelWidth, 
        UI::PanelHeight, 
        UI::PanelColor
    );
    
    // title
    DrawText("CAR TELEMETRY", textX, textY, UI::FontSize, UI::HeaderColor);
    textY += UI::LineHeight + UI::SectionSpacing;
    
    // motion section
    textY = drawSection(textX, textY, "MOTION", {
        [&](int x, int y) { DrawText(TextFormat("Speed: %.2f km/h", car.speed * 3.6f), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Rotation: %.2f°", car.rotation * RAD2DEG), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Position: (%.1f, %.1f, %.1f)", 
                                    car.position.x, car.position.y, car.position.z), x, y, UI::FontSize, UI::TextColor); }
    });
    
    // engine section
    textY = drawSection(textX, textY, "ENGINE", {
        [&](int x, int y) { DrawText(TextFormat("Engine Speed: %.0f RPM", car.engineSpeed), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Engine Torque: %.1f Nm", 
                                    car.getEngineTorque(car.throttle, car.engineSpeed)), x, y, UI::FontSize, UI::TextColor); }
    });
    
    // controls section
    textY = drawSection(textX, textY, "CONTROLS", {
        [&](int x, int y) { DrawText(TextFormat("Throttle: %.2f", car.throttle), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Brake: %.2f", car.brake), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Steering Angle: %.2f°", car.steeringAngle * RAD2DEG), x, y, UI::FontSize, UI::TextColor); }
    });
    
    // physics section
    textY = drawSection(textX, textY, "PHYSICS", {
        [&](int x, int y) { DrawText(TextFormat("Accel X: %.2f m/s²", car.acceleration.x), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Velocity X: %.2f m/s", car.velocity.x), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Velocity Z: %.2f m/s", car.velocity.z), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Slip Ratio: %.3f", car.slipRatio), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Force Generated by Wheels: %.2f N", car.longitudinalForce), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Net Force on Car Body: %.2f N", car.netForce), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Drag Force: %.2f N", car.dragForce), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Rolling Resistance: %.2f N", car.rollingResistance), x, y, UI::FontSize, UI::TextColor); }
    });
    
    // lateral dynamics
    textY = drawSection(textX, textY, "LATERAL DYNAMICS", {
        [&](int x, int y) { DrawText(TextFormat("Lateral Velocity: %.2f m/s", car.lateralVelocity), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Yaw Rate: %.2f rad/s", car.yawRate), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Slip Angle Front: %.2f°", car.slipAngleFront * RAD2DEG), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Slip Angle Rear: %.2f°", car.slipAngleRear * RAD2DEG), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Lateral Force Front: %.2f N", car.lateralForceFront), x, y, UI::FontSize, UI::TextColor); },
        [&](int x, int y) { DrawText(TextFormat("Lateral Force Rear: %.2f N", car.lateralForceRear), x, y, UI::FontSize, UI::TextColor); }
    });
}

void Renderer::drawInstructions() {
    
}

int Renderer::drawSection(int x, int y, const char* title, 
                         const std::vector<std::function<void(int, int)>>& drawFuncs) {
    // section header
    DrawText(title, x, y, UI::SmallFontSize, UI::SectionColor);
    y += UI::LineHeight;
    
    // call each drawing function with current pos
    for (const auto& drawFunc : drawFuncs) {
        drawFunc(x, y);
        y += UI::LineHeight;
    }
    
    // spacing after section
    y += UI::SectionSpacing;
    return y;
}

} // namespace CarGame
