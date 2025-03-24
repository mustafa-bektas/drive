#include "../include/car.h"
#include "../include/camera.h"
#include "../include/rendering.h"
#include "../include/input_handler.h"
#include "rl_agent.h"     // Add the RL agent header
#include "episode_manager.h"  // Add episode manager
#include <memory>
#include <chrono>

using namespace CarGame;

int main(void) {
    // Initialization
    const int screenWidth = 1920;
    const int screenHeight = 1080;
    InitWindow(screenWidth, screenHeight, "Car Game with RL");

    Vector3 startPosition = { 0.0f, 0.5f, 0.0f };
    Car car(startPosition);
    GameCamera camera;
    Renderer renderer;
    InputHandler inputHandler;
    
    // Initialize RL components
    RLAgent agent(0.1f, 0.9f, 0.5f);  // learning rate, discount factor, exploration rate
    const float targetSpeed = 8.33f;   // m/s (about 72 km/h)
    EpisodeManager episodeManager(30.0f, targetSpeed);  // 30 second episodes
    
    // Training mode flag (toggle with T key)
    bool trainingMode = true;
    bool lastKeyT = false;
    
    // Initialize renderer
    renderer.initialize(car);

    Vector3 floorPosition = { 0.0f, 0.0f, 0.0f };

    SetTargetFPS(60);
    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastSaveTime = startTime;

    // Main game loop
    while (!WindowShouldClose()) {
        float deltaTime = GetFrameTime();
        
        // Check for training mode toggle
        bool currentKeyT = IsKeyDown(KEY_T);
        if (currentKeyT && !lastKeyT) {
            trainingMode = !trainingMode;
            printf("Training mode: %s\n", trainingMode ? "ON" : "OFF");
        }
        lastKeyT = currentKeyT;
        
        if (trainingMode) {
            // RL agent controls the car
            
            // 1. Get current state
            std::vector<float> state = {car.speed};
            
            // 2. Get action from agent
            std::vector<float> action = agent.getAction(state);
            
            // 3. Apply action to car
            car.throttle = action[0];
            car.brake = action[1];
            
            // 4. Update car physics
            car.update(deltaTime);
            
            // 5. Get new state and calculate reward
            std::vector<float> nextState = {car.speed};
            float reward = -std::abs(car.speed - targetSpeed);
            
            // 6. Update agent
            agent.updateQValues(state, action, reward, nextState);
            
            // 7. Update episode manager
            episodeManager.update(deltaTime, car, agent);
            
            // 8. Update renderer stats
            renderer.updateRLStats(reward, car.speed - targetSpeed);
            
            // Auto-save every 5 minutes
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsedSecs = std::chrono::duration_cast<std::chrono::seconds>(
                currentTime - lastSaveTime).count();
                
            if (elapsedSecs > 300) {  // 5 minutes
                agent.saveModel("rl_model.dat");
                episodeManager.saveStats("training_stats.csv");
                lastSaveTime = currentTime;
                printf("Auto-saved model and stats\n");
            }
            
            // Start a new episode if needed
            if (episodeManager.isEpisodeComplete()) {
                renderer.newEpisode();
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
        
        // Draw RL stats if in training mode
        if (trainingMode) {
            renderer.drawRLStats(car, agent, episodeManager.getCurrentReward(), targetSpeed);
        }
        
        // Draw mode indicator
        DrawText(trainingMode ? "TRAINING MODE (Press T to toggle)" : 
                              "MANUAL MODE (Press T to toggle)",
                 GetScreenWidth() - 400, GetScreenHeight() - 60, 20, trainingMode ? RED : GREEN);
        
        EndDrawing();
    }

    // Save the final model and stats
    agent.saveModel("rl_model_final.dat");
    episodeManager.saveStats("training_stats_final.csv");

    CloseWindow();
    return 0;
}