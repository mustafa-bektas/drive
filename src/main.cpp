#include "raylib.h"
#include "car.h"
#include "camera.h"
#include "rendering.h"
#include "dqn_environment.h"
#include "dqn_agent.h"
#include "model_loader.h"
#include <memory>
#include <string>

using namespace CarGame;

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string modelFile = "models/best_model_for_cpp.txt";
    float targetSpeed = 50.0f / 3.6f;  // 50 km/h in m/s
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelFile = argv[++i];
        } else if (arg == "--target-speed" && i + 1 < argc) {
            targetSpeed = std::stof(argv[++i]) / 3.6f;  // Convert km/h to m/s
        }
    }
    
    // Initialization
    const int screenWidth = 1280;
    const int screenHeight = 720;
    InitWindow(screenWidth, screenHeight, "Car Game - DQN Speed Control");

    // Create DQN environment
    DQNEnvironment::Config envConfig;
    envConfig.targetSpeed = targetSpeed;
    envConfig.maxEpisodeSteps = 500;
    
    DQNEnvironment env(envConfig);
    
    // Create DQN agent and load model
    DQNAgent::Config agentConfig;
    agentConfig.stateSize = env.getStateSize();
    agentConfig.actionSize = env.getActionSize();
    
    DQNAgent agent(agentConfig);
    bool modelLoaded = agent.loadModel(modelFile);
    
    if (!modelLoaded) {
        DrawText("Failed to load model!", 400, 300, 20, RED);
    }
    
    // Create rendering components
    GameCamera camera;
    Renderer renderer;
    renderer.initialize(env.getCar());
    
    Vector3 floorPosition = { 0.0f, 0.0f, 0.0f };
    
    // Setup simulation
    std::vector<float> state = env.reset();
    float totalReward = 0.0f;
    int stepCount = 0;
    
    // Simulation control
    bool paused = false;
    float simulationSpeed = 1.0f;
    
    SetTargetFPS(60);

    // Main game loop
    while (!WindowShouldClose()) {
        float deltaTime = GetFrameTime();
        
        // Process user input for controlling the simulation
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        
        if (IsKeyPressed(KEY_R)) {
            // Reset environment
            state = env.reset();
            totalReward = 0.0f;
            stepCount = 0;
        }
        
        if (IsKeyPressed(KEY_UP)) {
            simulationSpeed *= 1.5f;
        }
        
        if (IsKeyPressed(KEY_DOWN)) {
            simulationSpeed /= 1.5f;
            if (simulationSpeed < 0.1f) simulationSpeed = 0.1f;
        }
        
        // Run simulation steps
        if (!paused) {
            int stepsThisFrame = std::max(1, static_cast<int>(simulationSpeed));
            
            for (int i = 0; i < stepsThisFrame; i++) {
                // Agent selects action based on current state
                DQNEnvironment::Action action = agent.selectAction(state, false);
                
                // Environment step
                std::tuple<std::vector<float>, float, bool> stepResult = env.step(action);
                std::vector<float> nextState = std::get<0>(stepResult);
                float reward = std::get<1>(stepResult);
                bool done = std::get<2>(stepResult);
                
                // Update metrics
                totalReward += reward;
                stepCount++;
                
                // Update state for next iteration
                state = nextState;
                
                // If episode is done, reset environment
                if (done) {
                    // Display episode summary
                    printf("Episode finished. Steps: %d, Total Reward: %.2f\n", 
                           stepCount, totalReward);
                    
                    // Reset for next episode
                    state = env.reset();
                    totalReward = 0.0f;
                    stepCount = 0;
                }
            }
        }
        
        // Update camera to follow the car
        camera.update(env.getCar());

        // Render
        BeginDrawing();
            ClearBackground(RAYWHITE);
            
            // Draw 3D scene with the car
            renderer.drawScene(camera, env.getCar(), floorPosition);
            
            // Draw simulation info
            DrawRectangle(10, 10, 250, 160, {200, 200, 200, 180});
            DrawText("DQN Speed Control", 20, 20, 20, DARKBLUE);
            DrawText(TextFormat("Target: %.1f km/h", env.getTargetSpeed() * 3.6f), 
                     20, 50, 18, DARKGREEN);
            DrawText(TextFormat("Current: %.1f km/h", env.getCar().speed * 3.6f), 
                     20, 75, 18, RED);
            DrawText(TextFormat("Reward: %.2f", totalReward), 
                     20, 100, 18, BLACK);
            DrawText(TextFormat("Steps: %d", stepCount), 
                     20, 125, 18, BLACK);
            DrawText(TextFormat("Speed: %.1fx %s", simulationSpeed, 
                     paused ? "[PAUSED]" : ""), 20, 150, 18, DARKGRAY);
            
            // Draw simulation control instructions
            DrawText("SPACE: Pause/Resume", 20, GetScreenHeight() - 100, 20, DARKGRAY);
            DrawText("R: Reset Simulation", 20, GetScreenHeight() - 75, 20, DARKGRAY);
            DrawText("UP/DOWN: Change Speed", 20, GetScreenHeight() - 50, 20, DARKGRAY);
            DrawText("ESC: Exit", 20, GetScreenHeight() - 25, 20, DARKGRAY);
            
        EndDrawing();
    }

    CloseWindow();

    return 0;
}