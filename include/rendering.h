#pragma once

#include "raylib.h"
#include "car.h"
#include "camera.h"
#include <vector>
#include <functional>
#include "rl_agent.h"

namespace CarGame {

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    void initialize(const Car& car);
    
    void drawScene(const GameCamera& camera, const Car& car, const Vector3& floorPosition);
    void drawRLStats(const Car& car, const RLAgent& agent, float reward, float targetSpeed);
    void updateRLStats(float reward, float speedError);
    void newEpisode();

private:
    void draw3DScene(const GameCamera& camera, const Car& car, const Vector3& floorPosition);
    
    void drawTelemetryPanel(const Car& car);
    void drawInstructions();
    
    int drawSection(int x, int y, const char* title, 
                   const std::vector<std::function<void(int, int)>>& drawFuncs);
    
    Model carModel;

    // RL tracking data
    struct RLStats {
        int episode = 0;
        float cumulativeReward = 0.0f;
        float averageReward = 0.0f;
        float bestReward = -std::numeric_limits<float>::max();
        std::vector<float> rewardHistory;
        std::vector<float> speedErrorHistory;
    };
    
    RLStats rlStats;
};

} // namespace CarGame