#include "../include/parallel_rl_agent.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <random>

namespace fs = std::filesystem;

namespace CarGame {

ParallelRLAgent::ParallelRLAgent(int instanceId, const std::string& sharedDirectory,
                              float learningRate, float discountFactor, float explorationRate)
    : RLAgent(learningRate, discountFactor, explorationRate),
      instanceId(instanceId),
      sharedDirectory(sharedDirectory),
      isRunning(false) {
    
    // Create shared directory if it doesn't exist
    if (!fs::exists(sharedDirectory)) {
        fs::create_directories(sharedDirectory);
        std::cout << "Created shared directory: " << sharedDirectory << std::endl;
    }
    
    // Start sharing thread
    startSharingThread();
}

ParallelRLAgent::~ParallelRLAgent() {
    stopSharingThread();
}

void ParallelRLAgent::startSharingThread() {
    if (!isRunning) {
        isRunning = true;
        sharingThread = std::thread(&ParallelRLAgent::sharingThreadFunction, this);
        std::cout << "Instance " << instanceId << ": Sharing thread started" << std::endl;
    }
}

void ParallelRLAgent::stopSharingThread() {
    if (isRunning) {
        isRunning = false;
        if (sharingThread.joinable()) {
            sharingThread.join();
        }
        std::cout << "Instance " << instanceId << ": Sharing thread stopped" << std::endl;
    }
}

void ParallelRLAgent::sharingThreadFunction() {
    // Random delay to avoid all instances trying to read/write simultaneously
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> delay_dist(1000, 5000);
    
    while (isRunning) {
        try {
            // Export our Q-table periodically
            {
                std::lock_guard<std::mutex> lock(qTableMutex);
                std::string filename = sharedDirectory + "/qtable_instance_" + 
                                      std::to_string(instanceId) + ".bin";
                exportQTable(filename);
            }
            
            // Check for other Q-tables and merge them
            checkAndLoadSharedQTables();
            
            // Random sleep between 5-15 seconds
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(gen)));
        } 
        catch (const std::exception& e) {
            std::cerr << "Instance " << instanceId << ": Error in sharing thread: " 
                     << e.what() << std::endl;
        }
    }
}

void ParallelRLAgent::trainFromReplay(int batchSize) {
    std::lock_guard<std::mutex> lock(qTableMutex);
    // Call the base class implementation
    RLAgent::trainFromReplay(batchSize);
}

void ParallelRLAgent::exportQTable(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Instance " << instanceId << ": Unable to open file for saving Q-table: " 
                 << filename << std::endl;
        return;
    }
    
    // Save Q-table size
    size_t stateCount = qTable.size();
    file.write(reinterpret_cast<const char*>(&stateCount), sizeof(size_t));
    
    // Save each state and its actions
    for (const auto& statePair : qTable) {
        // Write state key
        size_t stateKeyLength = statePair.first.length();
        file.write(reinterpret_cast<const char*>(&stateKeyLength), sizeof(size_t));
        file.write(statePair.first.c_str(), stateKeyLength);
        
        // Write number of actions for this state
        size_t actionCount = statePair.second.size();
        file.write(reinterpret_cast<const char*>(&actionCount), sizeof(size_t));
        
        // Write each action and its Q-value
        for (const auto& actionPair : statePair.second) {
            // Write action key
            size_t actionKeyLength = actionPair.first.length();
            file.write(reinterpret_cast<const char*>(&actionKeyLength), sizeof(size_t));
            file.write(actionPair.first.c_str(), actionKeyLength);
            
            // Write Q-value
            file.write(reinterpret_cast<const char*>(&actionPair.second), sizeof(float));
        }
    }
    
    file.close();
    std::cout << "Instance " << instanceId << ": Q-table exported to " << filename << std::endl;
}

void ParallelRLAgent::importQTable(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Instance " << instanceId << ": Unable to open file for loading Q-table: " 
                 << filename << std::endl;
        return;
    }
    
    std::map<std::string, std::map<std::string, float>> importedQTable;
    
    // Load Q-table size
    size_t stateCount;
    file.read(reinterpret_cast<char*>(&stateCount), sizeof(size_t));
    
    // Load each state and its actions
    for (size_t i = 0; i < stateCount; i++) {
        // Read state key
        size_t stateKeyLength;
        file.read(reinterpret_cast<char*>(&stateKeyLength), sizeof(size_t));
        
        std::string stateKey(stateKeyLength, ' ');
        file.read(&stateKey[0], stateKeyLength);
        
        // Create new map for this state
        importedQTable[stateKey] = std::map<std::string, float>();
        
        // Read number of actions for this state
        size_t actionCount;
        file.read(reinterpret_cast<char*>(&actionCount), sizeof(size_t));
        
        // Read each action and its Q-value
        for (size_t j = 0; j < actionCount; j++) {
            // Read action key
            size_t actionKeyLength;
            file.read(reinterpret_cast<char*>(&actionKeyLength), sizeof(size_t));
            
            std::string actionKey(actionKeyLength, ' ');
            file.read(&actionKey[0], actionKeyLength);
            
            // Read Q-value
            float qValue;
            file.read(reinterpret_cast<char*>(&qValue), sizeof(float));
            
            // Store in imported Q-table
            importedQTable[stateKey][actionKey] = qValue;
        }
    }
    
    file.close();
    
    // Merge the imported Q-table with our current Q-table
    mergeQTable(importedQTable);
    
    std::cout << "Instance " << instanceId << ": Q-table imported from " << filename << std::endl;
}

void ParallelRLAgent::mergeQTable(const std::map<std::string, std::map<std::string, float>>& otherQTable) {
    // For each state in the other Q-table
    for (const auto& statePair : otherQTable) {
        const std::string& stateKey = statePair.first;
        
        // If we don't have this state, add it
        if (qTable.find(stateKey) == qTable.end()) {
            qTable[stateKey] = std::map<std::string, float>();
        }
        
        // For each action in this state
        for (const auto& actionPair : statePair.second) {
            const std::string& actionKey = actionPair.first;
            float otherQValue = actionPair.second;
            
            // If we don't have this action, add it
            if (qTable[stateKey].find(actionKey) == qTable[stateKey].end()) {
                qTable[stateKey][actionKey] = otherQValue;
            } 
            // Otherwise, average the Q-values (simple ensemble method)
            else {
                float ourQValue = qTable[stateKey][actionKey];
                qTable[stateKey][actionKey] = (ourQValue + otherQValue) / 2.0f;
            }
        }
    }
}

void ParallelRLAgent::checkAndLoadSharedQTables() {
    try {
        for (const auto& entry : fs::directory_iterator(sharedDirectory)) {
            std::string filename = entry.path().string();
            
            // Skip our own Q-table
            if (filename.find("qtable_instance_" + std::to_string(instanceId)) != std::string::npos) {
                continue;
            }
            
            // Skip non-Q-table files
            if (filename.find("qtable_instance_") == std::string::npos) {
                continue;
            }
            
            // Import the Q-table
            {
                std::lock_guard<std::mutex> lock(qTableMutex);
                importQTable(filename);
            }
            
            // Optional: Move processed file to avoid reprocessing
            // fs::rename(filename, filename + ".processed");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Instance " << instanceId << ": Error checking shared Q-tables: " 
                 << e.what() << std::endl;
    }
}

} // namespace CarGame