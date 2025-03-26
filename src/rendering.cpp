#include "../include/rendering.h"
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
        
        // road with lanes
        const float roadWidth = 20.0f;
        const float laneWidth = 10.0f;
        
        // road surface
        DrawCube({0, 0.01f, 0}, roadWidth, 0.01f, groundSize, DARKGRAY);
        
        // lane center (dashed)
        for (int i = -groundSize/2; i < groundSize/2; i += 5) {
            DrawCube({0, 0.02f, float(i)}, 0.5f, 0.01f, 2.0f, YELLOW);
        }
        
        // lane edges
        DrawCube({-laneWidth/2, 0.02f, 0}, 0.3f, 0.01f, groundSize, WHITE);
        DrawCube({laneWidth/2, 0.02f, 0}, 0.3f, 0.01f, groundSize, WHITE);
        
        // road edges
        DrawCube({-roadWidth/2, 0.02f, 0}, 0.5f, 0.01f, groundSize, RED);
        DrawCube({roadWidth/2, 0.02f, 0}, 0.5f, 0.01f, groundSize, RED);
        
        // grid for reference
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
        
        // trees on road sides
        for (int i = -groundSize/2; i <= groundSize/2; i += 20) {
            // left trees
            DrawCylinder({-15, 0, float(i)}, 0.5f, 0.5f, 5.0f, 8, BROWN);
            DrawSphere({-15, 5.0f, float(i)}, 3.0f, DARKGREEN);
            
            // right trees
            DrawCylinder({15, 0, float(i)}, 0.5f, 0.5f, 5.0f, 8, BROWN);
            DrawSphere({15, 5.0f, float(i)}, 3.0f, DARKGREEN);
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