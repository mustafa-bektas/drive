#include "../include/car.h"
#include "../include/camera.h"
#include "../include/rendering.h"
#include "../include/input_handler.h"
#include "../include/rl_agent.h"
#include "../include/enhanced_episode_manager.h"
#include "enhanced_rendering.cpp"
#include <memory>
#include <chrono>

using namespace CarGame;

int main(void) {
    // Initialization
    const int screenWidth = 1920;
    const int screenHeight = 1080;
    InitWindow(screenWidth, screenHeight, "Enhanced Car Game with Fast RL Learning");

    Vector3 startPosition = { 0.0f, 0.5f, 0.0f };
    Car car(startPosition);
    GameCamera camera;
    Renderer renderer;
    InputHandler inputHandler;
    
    // Initialize enhanced RL components with SUPER aggressive parameters
    RLAgent agent(0.8f, 0.95f, 1.0f);  // Very high learning rate, high discount factor, full exploration
    const float targetSpeed = 8.33f;    // m/s (about 30 km/h)
    
    // Ultra-short episodes (5 seconds) for much faster learning cycles
    EnhancedEpisodeManager episodeManager(5.0f, targetSpeed, 1, 64);
    
    // Training mode flag (toggle with T key)
    bool trainingMode = true;
    bool lastKeyT = false;
    
    // Initialize renderer
    renderer.initialize(car);

    Vector3 floorPosition = { 0.0f, 0.0f, 0.0f };

    SetTargetFPS(60);
    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastSaveTime = startTime;
    
    // Capture time for each frame 
    float lastFrameTime = GetTime();

    // Main game loop
    while (!WindowShouldClose()) {
        float currentTime = GetTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        
        // Cap deltaTime to avoid physics instability
        if (deltaTime > 0.1f) deltaTime = 0.1f;
        
        // Check for training mode toggle
        bool currentKeyT = IsKeyDown(KEY_T);
        if (currentKeyT && !lastKeyT) {
            trainingMode = !trainingMode;
            printf("Training mode: %s\n", trainingMode ? "ON" : "OFF");
        }
        lastKeyT = currentKeyT;
        
        if (trainingMode) {
            // Enhanced RL agent controls the car
            episodeManager.update(deltaTime, car, agent);
            
            // Update car physics
            car.update(deltaTime);
            
            // Auto-save every 5 minutes
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsedSecs = std::chrono::duration_cast<std::chrono::seconds>(
                currentTime - lastSaveTime).count();
                
            if (elapsedSecs > 300) {  // 5 minutes
                agent.saveModel("rl_model_enhanced.dat");
                episodeManager.saveStats("training_stats_enhanced.csv");
                lastSaveTime = currentTime;
                printf("Auto-saved model and stats\n");
            }
        } else {
            // Manual control mode
            inputHandler.processInput(car, deltaTime);
            car.update(deltaTime);
        }
        
        // Always update camera
        camera.update(car);

        // Render
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        // Draw main scene
        renderer.drawScene(camera, car, floorPosition);
        
        // Draw enhanced RL visualization if in training mode
        if (trainingMode) {
            drawEnhancedRLStats(car, agent, episodeManager);
            drawLearningGraphs(episodeManager, agent);
            drawActionSpace(agent, car);
        }
        
        // Draw mode indicator
        DrawText(trainingMode ? "ENHANCED TRAINING MODE (Press T to toggle)" : 
                              "MANUAL MODE (Press T to toggle)",
                 GetScreenWidth() - 400, GetScreenHeight() - 60, 20, trainingMode ? RED : GREEN);
        
        EndDrawing();
    }

    // Save the final model and stats
    agent.saveModel("rl_model_enhanced_final.dat");
    episodeManager.saveStats("training_stats_enhanced_final.csv");

    CloseWindow();
    return 0;
}