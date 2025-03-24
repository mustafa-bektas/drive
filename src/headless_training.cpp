#include "../include/dqn_environment.h"
#include "../include/dqn_agent.h"
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace CarGame;

// Simple model saving/loading
void saveModelWeights(const NeuralNetwork& network, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    
    // Get network weights and biases
    auto weights = network.getAllWeights();
    auto biases = network.getAllBiases();
    
    // Save layer dimensions
    auto layerSizes = network.getLayerSizes();
    size_t numLayers = layerSizes.size();
    file.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));
    
    for (const auto& size : layerSizes) {
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }
    
    // Save weights
    for (const auto& layer : weights) {
        size_t numNeurons = layer.size();
        file.write(reinterpret_cast<const char*>(&numNeurons), sizeof(numNeurons));
        
        for (const auto& neuron : layer) {
            size_t numWeights = neuron.size();
            file.write(reinterpret_cast<const char*>(&numWeights), sizeof(numWeights));
            
            file.write(reinterpret_cast<const char*>(neuron.data()), 
                       static_cast<std::streamsize>(numWeights * sizeof(float)));
        }
    }
    
    // Save biases
    for (const auto& layer : biases) {
        size_t numBiases = layer.size();
        file.write(reinterpret_cast<const char*>(&numBiases), sizeof(numBiases));
        file.write(reinterpret_cast<const char*>(layer.data()), 
                   static_cast<std::streamsize>(numBiases * sizeof(float)));
    }
    
    file.close();
}

// Helper functions for time tracking and display
std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

void printProgressBar(int progress, int total, int width = 50) {
    float percentage = static_cast<float>(progress) / total;
    int filled = static_cast<int>(width * percentage);
    
    std::cout << "\r[";
    for (int i = 0; i < width; ++i) {
        if (i < filled) std::cout << "=";
        else std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(percentage * 100.0f) << "% "
              << progress << "/" << total << "  " << std::flush;
}

int main(int argc, char* argv[]) {
    std::cout << "===========================================" << std::endl;
    std::cout << "  Car Game - Headless DQN Training Mode   " << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Training configuration
    int numEpisodes = 1000;
    int saveInterval = 50;
    int reportInterval = 10;
    float targetSpeed = 50.0f / 3.6f;  // 50 km/h in m/s
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (i + 1 < argc) {
            if (arg == "--episodes") {
                numEpisodes = std::stoi(argv[i+1]);
            } else if (arg == "--save-interval") {
                saveInterval = std::stoi(argv[i+1]);
            } else if (arg == "--report-interval") {
                reportInterval = std::stoi(argv[i+1]);
            } else if (arg == "--target-speed") {
                targetSpeed = std::stof(argv[i+1]) / 3.6f;  // Convert km/h to m/s
            }
        }
    }
    
    std::cout << "Training Configuration:" << std::endl;
    std::cout << "  Number of Episodes:  " << numEpisodes << std::endl;
    std::cout << "  Save Interval:       " << saveInterval << " episodes" << std::endl;
    std::cout << "  Target Speed:        " << (targetSpeed * 3.6f) << " km/h" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    
    // Setup training environment
    DQNEnvironment::Config envConfig;
    envConfig.targetSpeed = targetSpeed;
    envConfig.maxEpisodeSteps = 1000;  // Longer episodes for better learning
    
    DQNEnvironment env(envConfig);
    
    // Create DQN agent
    DQNAgent::Config agentConfig;
    agentConfig.stateSize = env.getStateSize();
    agentConfig.actionSize = env.getActionSize();
    agentConfig.hiddenSize1 = 64;
    agentConfig.hiddenSize2 = 32;
    agentConfig.batchSize = 64;          // Larger batch size for faster learning
    agentConfig.replayBufferSize = 50000; // Larger buffer size
    agentConfig.epsilonStart = 1.0f;
    agentConfig.epsilonMin = 0.05f;
    agentConfig.epsilonDecay = 0.995f;
    
    DQNAgent agent(agentConfig);
    
    // Track training metrics
    std::vector<float> episodeRewards;
    std::vector<float> episodeSpeeds;
    float totalTrainingReward = 0.0f;
    
    // Main training loop
    auto trainingStartTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "Starting training..." << std::endl;
    
    for (int episode = 0; episode < numEpisodes; ++episode) {
        // Reset environment
        std::vector<float> state = env.reset();
        float episodeReward = 0.0f;
        float avgSpeed = 0.0f;
        int episodeSteps = 0;
        bool done = false;
        
        // Episode loop
        while (!done) {
            // Select action and step environment
            DQNEnvironment::Action action = agent.selectAction(state, true);
            
            // Environment step
            std::tuple<std::vector<float>, float, bool> stepResult = env.step(action);
            std::vector<float> nextState = std::get<0>(stepResult);
            float reward = std::get<1>(stepResult);
            done = std::get<2>(stepResult);
            
            // Update agent
            agent.addExperience(state, action, reward, nextState, done);
            
            // Train the agent
            if (agent.canTrain()) {
                agent.trainNetwork();
            }
            
            // Update metrics
            episodeReward += reward;
            avgSpeed += env.getCar().speed;
            episodeSteps++;
            
            // Update state for next step
            state = nextState;
        }
        
        // Calculate episode statistics
        avgSpeed = avgSpeed / episodeSteps * 3.6f;  // Convert to km/h
        totalTrainingReward += episodeReward;
        episodeRewards.push_back(episodeReward);
        episodeSpeeds.push_back(avgSpeed);
        
        // Report progress
        if ((episode + 1) % reportInterval == 0 || episode == 0 || episode == numEpisodes - 1) {
            float avgReward = totalTrainingReward / (episode + 1);
            std::cout << "Episode " << (episode + 1) << "/" << numEpisodes 
                      << " | Reward: " << std::fixed << std::setprecision(2) << episodeReward 
                      << " | Avg Reward: " << avgReward
                      << " | Avg Speed: " << avgSpeed << " km/h"
                      << " | Epsilon: " << agent.getExplorationRate()
                      << " | Steps: " << episodeSteps << std::endl;
        } else {
            // Show a simple progress bar
            printProgressBar(episode + 1, numEpisodes);
        }
        
        // Save model periodically
        if ((episode + 1) % saveInterval == 0 || episode == numEpisodes - 1) {
            std::string modelFilename = "model_" + getTimestamp() + "_e" + std::to_string(episode + 1) + ".model";
            saveModelWeights(*agent.getQNetwork(), modelFilename);
            std::cout << "Model saved to " << modelFilename << std::endl;
        }
    }
    
    // Print training summary
    auto trainingEndTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        trainingEndTime - trainingStartTime).count();
    
    std::cout << "\n\nTraining Complete!" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Total training time: " << duration / 60 << " minutes, " 
              << duration % 60 << " seconds" << std::endl;
    std::cout << "Average reward: " << totalTrainingReward / numEpisodes << std::endl;
    
    // Save final model
    std::string finalModelFilename = "model_final_" + getTimestamp() + ".model";
    saveModelWeights(*agent.getQNetwork(), finalModelFilename);
    std::cout << "Final model saved to " << finalModelFilename << std::endl;
    
    return 0;
}