#include "raylib.h"
#include "car.h"
#include "camera.h"
#include "rendering.h"
#include "dqn_environment.h"
#include "dqn_agent.h"
#include "neural_network.h"
#include "model_loader.h"
#include "visualization_helper.h"
#include <memory>
#include <string>
#include <iostream>

using namespace CarGame;

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string modelFile = "models/best_model_for_cpp.txt";
    float targetSpeed = 50.0f / 3.6f;  // 50 km/h in m/s
    
    // Process command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelFile = argv[++i];
        } else if (arg == "--target-speed" && i + 1 < argc) {
            targetSpeed = std::stof(argv[++i]) / 3.6f;  // Convert km/h to m/s
        }
    }
    
    // Initialization
    const int screenWidth = 1920;
    const int screenHeight = 1080;
    InitWindow(screenWidth, screenHeight, "Car Game - DQN Speed Control Demo");
    SetExitKey(KEY_NULL); // Disable default ESC key exit to handle it manually
    
    // Create environment
    DQNEnvironment::Config envConfig;
    envConfig.targetSpeed = targetSpeed;
    envConfig.maxEpisodeSteps = 600;
    
    DQNEnvironment env(envConfig);
    
    // Create network and load weights
    DQNAgent::Config agentConfig;
    agentConfig.stateSize = env.getStateSize();
    agentConfig.actionSize = env.getActionSize();
    
    DQNAgent agent(agentConfig);
    bool modelLoaded = agent.loadModel(modelFile);
    
    if (!modelLoaded) {
        std::cout << "Failed to load model: " << modelFile << "\nWill run with random actions." << std::endl;
    } else {
        std::cout << "Successfully loaded model: " << modelFile << std::endl;
    }
    
    // Create rendering components
    GameCamera camera;
    Renderer renderer;
    renderer.initialize(env.getCar());
    
    // Create visualization helper
    VisualizationHelper visualizer(120); // 120 frames of history
    
    Vector3 floorPosition = {0.0f, 0.0f, 0.0f};
    
    // Reset the environment
    std::vector<float> state = env.reset();
    float totalReward = 0.0f;
    int step = 0;
    
    // Simulation control
    bool paused = false;
    bool fullscreen = false;
    float simulationSpeed = 1.0f;
    float lastUpdateTime = GetTime();
    
    SetTargetFPS(60);
    
    // Main game loop
    while (!WindowShouldClose() && !IsKeyPressed(KEY_ESCAPE)) {
        float currentTime = GetTime();
        float deltaTime = currentTime - lastUpdateTime;
        lastUpdateTime = currentTime;
        
        // Process input for simulation control
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        
        if (IsKeyPressed(KEY_PAGE_UP) || IsKeyPressed(KEY_UP)) {
            simulationSpeed *= 1.5f;
            if (simulationSpeed > 8.0f) simulationSpeed = 8.0f;
        }
        
        if (IsKeyPressed(KEY_PAGE_DOWN) || IsKeyPressed(KEY_DOWN)) {
            simulationSpeed /= 1.5f;
            if (simulationSpeed < 0.1f) simulationSpeed = 0.1f;
        }
        
        if (IsKeyPressed(KEY_R)) {
            // Reset environment
            state = env.reset();
            totalReward = 0.0f;
            step = 0;
            visualizer.reset();
        }
        
        if (IsKeyPressed(KEY_F)) {
            fullscreen = !fullscreen;
            ToggleFullscreen();
        }
        
        if (IsKeyPressed(KEY_TAB)) {
            visualizer.toggleUI();
        }
        
        // Run simulation if not paused
        if (!paused) {
            // Multiple simulation steps based on simulation speed
            int stepsThisFrame = std::max(1, static_cast<int>(simulationSpeed));
            
            for (int i = 0; i < stepsThisFrame; i++) {
                // Select action using the trained model
                DQNEnvironment::Action action;
                
                if (modelLoaded) {
                    action = agent.selectAction(state, false); // No exploration during demo
                } else {
                    // Random action if model not loaded
                    action = static_cast<DQNEnvironment::Action>(GetRandomValue(0, env.getActionSize() - 1));
                }
                
                // Take a step in the environment
                std::tuple<std::vector<float>, float, bool> result = env.step(action);
                std::vector<float> nextState = std::get<0>(result);
                float reward = std::get<1>(result);
                bool done = std::get<2>(result);
                
                // Update visualization metrics
                visualizer.updateMetrics(env.getCar(), reward, action, env.getTargetSpeed());
                
                totalReward += reward;
                step++;
                
                // Update state
                state = nextState;
                
                // Reset if done
                if (done) {
                    printf("Episode completed. Steps: %d, Total Reward: %.2f\n", step, totalReward);
                    state = env.reset();
                    totalReward = 0.0f;
                    step = 0;
                    visualizer.reset();
                }
            }
        }
        
        // Update camera to follow the car
        camera.update(env.getCar());
        
        // Render
        BeginDrawing();
            ClearBackground(RAYWHITE);
            
            // Draw 3D scene
            renderer.drawScene(camera, env.getCar(), floorPosition);
            
            // Draw visualization UI (now more compact)
            visualizer.drawUI(env.getCar(), totalReward, step, env.getTargetSpeed(), 
                            simulationSpeed, paused, modelLoaded);
            
        EndDrawing();
    }
    
    // Clean up before exit
    if (fullscreen) {
        ToggleFullscreen();
    }
    
    CloseWindow();
    
    return 0;
}