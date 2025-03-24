#include "raylib.h"
#include "car.h"
#include "camera.h"
#include "rendering.h"
#include "dqn_environment.h"
#include "dqn_agent.h"
#include "neural_network.h"
#include "model_loader.h"
#include <memory>
#include <string>
#include <iostream>

using namespace CarGame;

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string modelFile = "trained_model.txt";
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
    InitWindow(screenWidth, screenHeight, "Car Game - Trained DQN Demo");
    
    // Create environment
    DQNEnvironment::Config envConfig;
    envConfig.targetSpeed = targetSpeed;
    envConfig.maxEpisodeSteps = 600;
    
    DQNEnvironment env(envConfig);
    
    // Create network and load weights
    std::vector<int> layerSizes = {env.getStateSize(), 64, 32, env.getActionSize()};
    NeuralNetwork network(layerSizes);
    
    bool modelLoaded = ModelLoader::loadModelFromPython(modelFile, network);
    if (!modelLoaded) {
        std::cout << "Failed to load model. Running with random actions." << std::endl;
    }
    
    // Create rendering components
    GameCamera camera;
    Renderer renderer;
    renderer.initialize(env.getCar());
    
    Vector3 floorPosition = {0.0f, 0.0f, 0.0f};
    
    // Reset the environment
    std::vector<float> state = env.reset();
    float totalReward = 0.0f;
    int step = 0;
    
    // Simulation speed control
    float simulationSpeed = 1.0f;
    
    SetTargetFPS(60);
    
    // Main game loop
    while (!WindowShouldClose()) {
        // Process input for simulation control
        if (IsKeyPressed(KEY_PAGE_UP)) {
            simulationSpeed *= 1.5f;
        }
        if (IsKeyPressed(KEY_PAGE_DOWN)) {
            simulationSpeed /= 1.5f;
            if (simulationSpeed < 0.1f) simulationSpeed = 0.1f;
        }
        if (IsKeyPressed(KEY_R)) {
            // Reset environment
            state = env.reset();
            totalReward = 0.0f;
            step = 0;
        }
        
        // Multiple simulation steps based on simulation speed
        int stepsThisFrame = std::max(1, static_cast<int>(simulationSpeed));
        
        for (int i = 0; i < stepsThisFrame; i++) {
            // Select action using the trained model
            DQNEnvironment::Action action;
            
            if (modelLoaded) {
                // Convert state to tensor for neural network
                std::vector<float> qValues = network.forward(state);
                
                // Find action with highest Q-value
                int bestActionIndex = 0;
                float bestValue = qValues[0];
                
                for (size_t j = 1; j < qValues.size(); j++) {
                    if (qValues[j] > bestValue) {
                        bestValue = qValues[j];
                        bestActionIndex = j;
                    }
                }
                
                action = static_cast<DQNEnvironment::Action>(bestActionIndex);
            } else {
                // Random action if model not loaded
                action = static_cast<DQNEnvironment::Action>(GetRandomValue(0, env.getActionSize() - 1));
            }
            
            // Take a step in the environment
            std::tuple<std::vector<float>, float, bool> result = env.step(action);
            std::vector<float> nextState = std::get<0>(result);
            float reward = std::get<1>(result);
            bool done = std::get<2>(result);
            
            totalReward += reward;
            step++;
            
            // Update state
            state = nextState;
            
            // Reset if done
            if (done) {
                state = env.reset();
                totalReward = 0.0f;
                step = 0;
            }
        }
        
        // Update camera to follow the car
        camera.update(env.getCar());
        
        // Render
        BeginDrawing();
            ClearBackground(RAYWHITE);
            
            // Draw 3D scene
            renderer.drawScene(camera, env.getCar(), floorPosition);
            
            // Draw telemetry
            DrawRectangle(10, 10, 300, 160, {200, 200, 200, 180});
            DrawText("Trained DQN Demo", 20, 20, 20, DARKBLUE);
            
            DrawText(TextFormat("Target Speed: %.1f km/h", env.getTargetSpeed() * 3.6f), 
                     20, 50, 18, DARKGREEN);
            DrawText(TextFormat("Current Speed: %.1f km/h", env.getCar().speed * 3.6f), 
                     20, 75, 18, RED);
            DrawText(TextFormat("Reward: %.2f", totalReward), 
                     20, 100, 18, BLUE);
            DrawText(TextFormat("Step: %d", step), 
                     20, 125, 18, BLACK);
            DrawText(TextFormat("Simulation Speed: %.1fx", simulationSpeed), 
                     20, 150, 18, MAROON);
            
            // Draw controls
            DrawText("Controls:", 10, GetScreenHeight() - 110, 20, DARKGRAY);
            DrawText("R - Reset Simulation", 20, GetScreenHeight() - 80, 18, DARKGRAY);
            DrawText("PAGE UP/DOWN - Adjust Simulation Speed", 20, GetScreenHeight() - 55, 18, DARKGRAY);
            DrawText("ESC - Exit", 20, GetScreenHeight() - 30, 18, DARKGRAY);
            
        EndDrawing();
    }
    
    CloseWindow();
    
    return 0;
}