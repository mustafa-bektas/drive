#include "../include/car.h"
#include "../include/camera.h"
#include "../include/rendering.h"
#include "../include/input_handler.h"
#include "../include/parallel_rl_agent.h"
#include "../include/enhanced_episode_manager.h"
#include "enhanced_rendering.cpp"
#include <memory>
#include <chrono>
#include <string>
#include <cstring>

using namespace CarGame;

// Helper function to parse command line arguments
bool getCommandLineArg(int argc, char* argv[], const std::string& arg, std::string& value) {
    for (int i = 1; i < argc; i++) {
        std::string currentArg = argv[i];
        if (currentArg.find(arg + "=") == 0) {
            value = currentArg.substr(arg.length() + 1);
            return true;
        }
    }
    return false;
}

bool hasCommandLineFlag(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments for parallel operation
    int instanceId = 0;
    std::string sharedDir = "./shared_rl";
    bool headlessMode = false;
    
    // Get instance ID
    std::string instanceIdStr;
    if (getCommandLineArg(argc, argv, "--instance-id", instanceIdStr)) {
        instanceId = std::stoi(instanceIdStr);
    }
    
    // Get shared directory
    getCommandLineArg(argc, argv, "--shared-dir", sharedDir);
    
    // Check for headless mode
    headlessMode = hasCommandLineFlag(argc, argv, "--headless");
    
    printf("Starting CarGame instance %d (Shared dir: %s, Headless: %s)\n", 
           instanceId, sharedDir.c_str(), headlessMode ? "Yes" : "No");
    
    // Initialize window only in graphical mode
    const int screenWidth = 1280;
    const int screenHeight = 720;
    
    if (!headlessMode) {
        InitWindow(screenWidth, screenHeight, 
                  ("Car Game with Parallel RL - Instance " + std::to_string(instanceId)).c_str());
    }

    Vector3 startPosition = { 0.0f, 0.5f, 0.0f };
    Car car(startPosition);
    GameCamera camera;
    Renderer renderer;
    InputHandler inputHandler;
    
    // Initialize parallel RL agent
    ParallelRLAgent agent(instanceId, sharedDir, 0.9f, 0.95f, 1.0f);
    const float targetSpeed = 8.33f;   // m/s (about 30 km/h)
    
    // Longer episodes (15 seconds) for better evaluation
    EnhancedEpisodeManager episodeManager(15.0f, targetSpeed, 1, 64);
    
    // Always in training mode for parallel instances
    bool trainingMode = true;
    
    // Initialize renderer in graphical mode
    if (!headlessMode) {
        renderer.initialize(car);
    }

    Vector3 floorPosition = { 0.0f, 0.0f, 0.0f };

    if (!headlessMode) {
        SetTargetFPS(60);
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastSaveTime = startTime;
    auto lastStatsTime = startTime;
    
    // Capture time for each frame 
    float lastFrameTime = GetTime();
    
    // Training frames counter
    int frameCount = 0;
    int episodeCount = 0;

    // Main game loop
    while (!headlessMode ? !WindowShouldClose() : episodeCount < 1000) {
        float currentTime = GetTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        
        // Cap deltaTime to avoid physics instability
        if (deltaTime > 0.1f) deltaTime = 0.1f;
        
        // Run at higher speed in headless mode
        if (headlessMode) {
            deltaTime *= 4.0f; // 4x speedup in headless mode
        }
        
        // Enhanced RL agent controls the car
        episodeManager.update(deltaTime, car, agent);
        
        // If episode completed, count it
        if (episodeManager.isEpisodeComplete()) {
            episodeCount++;
        }
        
        // Update car physics
        car.update(deltaTime);
        
        // Auto-save every 5 minutes
        auto currentTimePoint = std::chrono::high_resolution_clock::now();
        auto elapsedSecs = std::chrono::duration_cast<std::chrono::seconds>(
            currentTimePoint - lastSaveTime).count();
            
        if (elapsedSecs > 300) {  // 5 minutes
            agent.saveModel("rl_model_instance_" + std::to_string(instanceId) + ".dat");
            episodeManager.saveStats("training_stats_instance_" + std::to_string(instanceId) + ".csv");
            lastSaveTime = currentTimePoint;
            printf("Instance %d: Auto-saved model and stats\n", instanceId);
        }
        
        // Print stats every 30 seconds
        auto statsSecs = std::chrono::duration_cast<std::chrono::seconds>(
            currentTimePoint - lastStatsTime).count();
            
        if (statsSecs > 30) {  // 30 seconds
            printf("Instance %d: Completed %d episodes, Current reward: %.2f, Success rate: %.2f%%\n",
                   instanceId, episodeCount, episodeManager.getCurrentReward(),
                   episodeManager.getSuccessRate() * 100.0f);
            lastStatsTime = currentTimePoint;
        }
        
        // Skip rendering in headless mode
        if (headlessMode) {
            continue;
        }
        
        // Always update camera in graphical mode
        camera.update(car);

        // Render
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        // Draw main scene
        renderer.drawScene(camera, car, floorPosition);
        
        // Draw enhanced RL visualization
        drawEnhancedRLStats(car, agent, episodeManager);
        drawLearningGraphs(episodeManager, agent);
        drawActionSpace(agent, car);
        
        // Show instance ID
        DrawText(TextFormat("PARALLEL RL INSTANCE %d", instanceId),
                 GetScreenWidth() - 400, GetScreenHeight() - 60, 20, RED);
        
        EndDrawing();
    }

    // Save the final model and stats
    agent.saveModel("rl_model_instance_" + std::to_string(instanceId) + "_final.dat");
    episodeManager.saveStats("training_stats_instance_" + std::to_string(instanceId) + "_final.csv");
    
    printf("Instance %d: Finished after %d episodes\n", instanceId, episodeCount);

    if (!headlessMode) {
        CloseWindow();
    }
    
    return 0;
}