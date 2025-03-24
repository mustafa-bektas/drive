#include "raylib.h"
#include "../include/car.h"
#include "../include/camera.h"
#include "../include/rendering.h"
#include "../include/dqn_environment.h"
#include "../include/dqn_agent.h"
#include "../include/dqn_visualizer.h"
#include <memory>
#include <string>

using namespace CarGame;

int main(void) {
    // Initialization
    const int screenWidth = 1280;
    const int screenHeight = 720;
    InitWindow(screenWidth, screenHeight, "Car Game - DQN Speed Control");

    // Create DQN environment
    DQNEnvironment::Config envConfig;
    envConfig.targetSpeed = 50.0f / 3.6f;  // 50 km/h in m/s
    envConfig.maxEpisodeSteps = 500;       // Longer episodes to see progress
    
    DQNEnvironment rlEnv(envConfig);
    
    // Create DQN agent
    DQNAgent::Config agentConfig;
    agentConfig.stateSize = rlEnv.getStateSize();
    agentConfig.actionSize = rlEnv.getActionSize();
    agentConfig.hiddenSize1 = 64;
    agentConfig.hiddenSize2 = 32;
    agentConfig.batchSize = 32;
    agentConfig.replayBufferSize = 10000;
    agentConfig.epsilonStart = 1.0f;
    agentConfig.epsilonMin = 0.1f;
    agentConfig.epsilonDecay = 0.995f;
    
    DQNAgent agent(agentConfig);
    DQNVisualizer visualizer(300);  // Keep 300 data points for visualization
    
    // Create rendering components
    GameCamera camera;
    Renderer renderer;
    renderer.initialize(rlEnv.getCar());  // Initialize with the RL environment's car
    
    Vector3 floorPosition = { 0.0f, 0.0f, 0.0f };
    
    // Setup RL training
    std::vector<float> state = rlEnv.reset();
    bool episodeFinished = false;
    bool trainingEnabled = true;
    bool singleStepMode = false;
    bool takeStep = false;
    int stepCount = 0;
    int trainingInterval = 4;  // Train every few steps for smoother visualization
    
    // Stats
    float totalReward = 0.0f;
    int episodeSteps = 0;
    float episodeStartTime = GetTime();
    
    SetTargetFPS(60);

    // Main game loop
    while (!WindowShouldClose()) {
        float deltaTime = GetFrameTime();
        
        // Process user input for controlling the simulation
        if (IsKeyPressed(KEY_SPACE)) {
            trainingEnabled = !trainingEnabled;  // Toggle training
        }
        
        if (IsKeyPressed(KEY_S)) {
            singleStepMode = !singleStepMode;    // Toggle single-step mode
        }
        
        if (IsKeyPressed(KEY_RIGHT) && singleStepMode) {
            takeStep = true;  // Take one step in single-step mode
        }
        
        if (IsKeyPressed(KEY_R)) {
            // Reset environment and start a new episode
            state = rlEnv.reset();
            episodeFinished = false;
            totalReward = 0.0f;
            episodeSteps = 0;
            episodeStartTime = GetTime();
            visualizer.startNewEpisode();
        }
        
        if (IsKeyPressed(KEY_T) && trainingEnabled) {
            // Manually trigger training
            if (agent.canTrain()) {
                agent.trainNetwork();
            }
        }
        
        // DQN training step
        if ((trainingEnabled && !singleStepMode) || (singleStepMode && takeStep)) {
            // Agent selects an action
            DQNEnvironment::Action action = agent.selectAction(state, true);  // true = exploration enabled
            visualizer.setLastAction(action);
            
            // Environment step - C++11 compatible way to handle tuple return
            std::tuple<std::vector<float>, float, bool> stepResult = rlEnv.step(action);
            std::vector<float> nextState = std::get<0>(stepResult);
            float reward = std::get<1>(stepResult);
            bool done = std::get<2>(stepResult);
            
            visualizer.setLastReward(reward);
            
            // Add to replay buffer
            agent.addExperience(state, action, reward, nextState, done);
            
            // Train the agent (periodically for smoother visualization)
            stepCount++;
            if (agent.canTrain() && stepCount % trainingInterval == 0) {
                agent.trainNetwork();
            }
            
            // Record data for visualization
            totalReward += reward;
            episodeSteps++;
            visualizer.recordEpisodeData(reward, rlEnv.getCar().speed, rlEnv.getTargetSpeed());
            
            // Update state for next iteration
            state = nextState;
            episodeFinished = done;
            
            // If episode is done, reset environment
            if (episodeFinished) {
                float episodeDuration = GetTime() - episodeStartTime;
                
                // Display episode summary
                printf("Episode %d finished. Steps: %d, Total Reward: %.2f, Duration: %.2f s\n",
                       visualizer.startNewEpisode(), episodeSteps, totalReward, episodeDuration);
                
                // Reset for next episode
                state = rlEnv.reset();
                episodeFinished = false;
                totalReward = 0.0f;
                episodeSteps = 0;
                episodeStartTime = GetTime();
            }
            
            takeStep = false;
        }
        
        // Update camera to follow the car
        camera.update(rlEnv.getCar());

        // Render
        BeginDrawing();
            ClearBackground(RAYWHITE);
            
            // Draw 3D scene with the car
            renderer.drawScene(camera, rlEnv.getCar(), floorPosition);
            
            // Draw RL visualization
            visualizer.draw(rlEnv, agent);
            
            // Draw simulation control instructions
            DrawText("SPACE: Toggle Training", 20, GetScreenHeight() - 100, 20, DARKGRAY);
            DrawText("S: Toggle Single-Step Mode", 20, GetScreenHeight() - 75, 20, DARKGRAY);
            DrawText("RIGHT ARROW: Step Forward (in Single-Step Mode)", 20, GetScreenHeight() - 50, 20, DARKGRAY);
            DrawText("R: Reset Episode", 20, GetScreenHeight() - 25, 20, DARKGRAY);
            
            // Draw training status
            DrawText(TextFormat("Training: %s", trainingEnabled ? "ON" : "OFF"), 
                     GetScreenWidth() - 200, GetScreenHeight() - 50, 20, 
                     trainingEnabled ? GREEN : RED);
            
            DrawText(TextFormat("Mode: %s", singleStepMode ? "STEP" : "CONTINUOUS"), 
                     GetScreenWidth() - 200, GetScreenHeight() - 25, 20, 
                     singleStepMode ? ORANGE : BLUE);
        EndDrawing();
    }

    CloseWindow();

    return 0;
}