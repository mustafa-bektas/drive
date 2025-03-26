#include "raylib.h"
#include "car.h"
#include "camera.h"
#include "rendering.h"
#include "dqn_environment.h"
#include "dqn_agent.h"
#include "lane_keeping_environment.h"
#include "lane_keeping_agent.h"
#include "neural_network.h"
#include "model_loader.h"
#include "visualization_helper.h"
#include <memory>
#include <string>
#include <iostream>

using namespace CarGame;

int main(int argc, char* argv[]) {
    // cli args
    std::string speedModelFile = "models/best_model_for_cpp.txt";
    std::string laneModelFile = "models/best_lane_keeping_model_for_cpp.txt";
    float targetSpeed = 50.0f / 3.6f;  // 50 km/h in m/s
    bool enableLaneKeeping = true;
    
    // handle args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--speed-model" && i + 1 < argc) {
            speedModelFile = argv[++i];
        } else if (arg == "--lane-model" && i + 1 < argc) {
            laneModelFile = argv[++i];
        } else if (arg == "--target-speed" && i + 1 < argc) {
            targetSpeed = std::stof(argv[++i]) / 3.6f;  // km/h to m/s
        } else if (arg == "--disable-lane-keeping") {
            enableLaneKeeping = false;
        }
    }
    
    // init
    const int screenWidth = 1280;
    const int screenHeight = 720;
    InitWindow(screenWidth, screenHeight, "Car Game - DQN Speed & Lane Keeping Demo");
    SetExitKey(KEY_NULL); // use custom exit handling
    
    // speed env
    DQNEnvironment::Config speedEnvConfig;
    speedEnvConfig.targetSpeed = targetSpeed;
    speedEnvConfig.maxEpisodeSteps = 1000;
    
    DQNEnvironment speedEnv(speedEnvConfig);
    
    // lane env
    LaneKeepingEnvironment::Config laneEnvConfig;
    laneEnvConfig.laneWidth = 10.0f;
    laneEnvConfig.maxLateralDeviation = 5.0f;
    laneEnvConfig.maxEpisodeSteps = 1000;
    
    LaneKeepingEnvironment laneEnv(laneEnvConfig);
    
    // speed agent
    DQNAgent::Config speedAgentConfig;
    speedAgentConfig.stateSize = speedEnv.getStateSize();
    speedAgentConfig.actionSize = speedEnv.getActionSize();
    
    DQNAgent speedAgent(speedAgentConfig);
    bool speedModelLoaded = speedAgent.loadModel(speedModelFile);
    
    if (!speedModelLoaded) {
        std::cout << "Failed to load speed control model: " << speedModelFile << std::endl;
        DrawText("Failed to load speed control model!", 400, 300, 20, RED);
        WaitTime(2.0); // show error for 2 secs
    } else {
        std::cout << "Successfully loaded speed control model: " << speedModelFile << std::endl;
    }
    
    // lane agent
    LaneKeepingAgent::Config laneAgentConfig;
    laneAgentConfig.stateSize = laneEnv.getStateSize();
    laneAgentConfig.actionSize = laneEnv.getActionSize();
    
    LaneKeepingAgent laneAgent(laneAgentConfig);
    bool laneModelLoaded = false;
    
    if (enableLaneKeeping) {
        laneModelLoaded = laneAgent.loadModel(laneModelFile);
        
        if (!laneModelLoaded) {
            std::cout << "Failed to load lane keeping model: " << laneModelFile << std::endl;
            DrawText("Failed to load lane keeping model", 400, 330, 20, RED);
            WaitTime(2.0); // show error for 2 secs
        } else {
            std::cout << "Successfully loaded lane keeping model: " << laneModelFile << std::endl;
        }
    }
    
    // rendering stuff
    GameCamera camera;
    Renderer renderer;
    renderer.initialize(speedEnv.getCar());
    
    // viz helper
    VisualizationHelper visualizer(120); // 120 frame history
    
    Vector3 floorPosition = {0.0f, 0.0f, 0.0f};
    
    // reset envs
    std::vector<float> speedState = speedEnv.reset();
    std::vector<float> laneState = laneEnv.reset(speedEnv.getCar());
    int step = 0;
    
    // sim controls
    bool paused = false;
    bool fullscreen = false;
    float simulationSpeed = 1.0f;
    float lastUpdateTime = GetTime();
    
    // lane keeping toggle
    bool laneKeepingActive = enableLaneKeeping && laneModelLoaded;
    
    SetTargetFPS(60);
    
    // main loop
    while (!WindowShouldClose() && !IsKeyPressed(KEY_ESCAPE)) {
        float currentTime = GetTime();
        float deltaTime = currentTime - lastUpdateTime;
        lastUpdateTime = currentTime;
        
        // sim control input
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        
        if (IsKeyPressed(KEY_PAGE_UP)) {
            simulationSpeed *= 1.5f;
            if (simulationSpeed > 8.0f) simulationSpeed = 8.0f;
        }
        
        if (IsKeyPressed(KEY_PAGE_DOWN)) {
            simulationSpeed /= 1.5f;
            if (simulationSpeed < 0.1f) simulationSpeed = 0.1f;
        }
        
        if (IsKeyPressed(KEY_R)) {
            // reset
            speedState = speedEnv.reset();
            laneState = laneEnv.reset(speedEnv.getCar());
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
        
        if (IsKeyPressed(KEY_L)) {
            // toggle lane keeping
            laneKeepingActive = !laneKeepingActive && enableLaneKeeping && laneModelLoaded;
            printf("Lane keeping: %s\n", laneKeepingActive ? "ON" : "OFF");
        }
        
        // run sim if not paused
        if (!paused) {
            // multi steps based on sim speed
            int stepsThisFrame = std::max(1, static_cast<int>(simulationSpeed));
            
            for (int i = 0; i < stepsThisFrame; i++) {
                Car& car = speedEnv.getCar();
                
                // lane keeping first if enabled
                if (laneKeepingActive) {
                    LaneKeepingEnvironment::Action laneAction = laneAgent.selectAction(laneState);
                    std::tuple<std::vector<float>, float, bool> laneResult = laneEnv.step(laneAction, car);
                    laneState = std::get<0>(laneResult);
                }
                
                // then speed control
                DQNEnvironment::Action speedAction = speedAgent.selectAction(speedState);
                std::tuple<std::vector<float>, float, bool> speedResult = speedEnv.step(speedAction);
                speedState = std::get<0>(speedResult);
                bool done = std::get<2>(speedResult);
                
                // update viz metrics
                visualizer.updateSpeedHistory(speedEnv.getCar(), speedEnv.getTargetSpeed());
                
                step++;
                
                // reset if done
                if (done) {
                    printf("Episode completed. Steps: %d\n", step);
                    
                    // reset for next ep
                    speedState = speedEnv.reset();
                    laneState = laneEnv.reset(speedEnv.getCar());
                    step = 0;
                    visualizer.reset();
                }
            }
        }
        
        // follow car with camera
        camera.update(speedEnv.getCar());
        
        // render
        BeginDrawing();
            ClearBackground(RAYWHITE);
            
            // 3d scene
            renderer.drawScene(camera, speedEnv.getCar(), floorPosition);
            
            // ui
            visualizer.drawUI(speedEnv.getCar(), speedEnv.getCar().speed, speedEnv.getTargetSpeed(), 
                            simulationSpeed, paused, speedModelLoaded);
            
            // lane viz
            if (enableLaneKeeping) {
                visualizer.drawLaneInfo(speedEnv.getCar(), laneEnvConfig.laneWidth, laneKeepingActive);
                
                // lane status
                const char* laneKeepingStatus = laneKeepingActive ? "Lane Keeping: ON (L to toggle)" : 
                                                              "Lane Keeping: OFF (L to toggle)";
                DrawText(laneKeepingStatus, 10, GetScreenHeight() - 50, 20, 
                        laneKeepingActive ? DARKGREEN : DARKGRAY);
            }
            
        EndDrawing();
    }
    
    // cleanup
    if (fullscreen) {
        ToggleFullscreen();
    }
    
    CloseWindow();
    
    return 0;
}