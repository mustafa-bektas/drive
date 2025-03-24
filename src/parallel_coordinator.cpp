#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <random>
#include <fstream>
#include <string>
#include <map>
#include <cstdlib>
#include <mutex>
#include <signal.h>

namespace fs = std::filesystem;

// Simple structure to hold program configuration
struct Config {
    int numInstances = 4;           // Number of parallel instances to run
    std::string sharedDir = "./shared_rl"; // Directory for sharing Q-tables
    std::string executablePath = "./CarGame"; // Path to the game executable
    int mergeInterval = 60;         // Seconds between global merges
    bool verbose = true;            // Print detailed output
};

// Global variables
std::atomic<bool> isRunning(true);
std::mutex consoleMutex;

// Signal handler for graceful shutdown
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received. Shutting down gracefully..." << std::endl;
    isRunning = false;
}

// Function to launch a CarGame instance
void launchGameInstance(int instanceId, const Config& config) {
    std::string cmd = config.executablePath + 
                     " --instance-id=" + std::to_string(instanceId) + 
                     " --shared-dir=" + config.sharedDir;
    
    if (config.verbose) {
        std::lock_guard<std::mutex> lock(consoleMutex);
        std::cout << "Launching instance " << instanceId << ": " << cmd << std::endl;
    }
    
    // Launch the process
    std::system(cmd.c_str());
    
    if (config.verbose) {
        std::lock_guard<std::mutex> lock(consoleMutex);
        std::cout << "Instance " << instanceId << " has terminated." << std::endl;
    }
}

// Merge all Q-tables into a master Q-table
void performGlobalMerge(const Config& config) {
    if (config.verbose) {
        std::lock_guard<std::mutex> lock(consoleMutex);
        std::cout << "Performing global Q-table merge..." << std::endl;
    }
    
    // Map to store the merged Q-table
    std::map<std::string, std::map<std::string, std::pair<float, int>>> mergedTable;
    
    // For each Q-table file in the shared directory
    for (const auto& entry : fs::directory_iterator(config.sharedDir)) {
        std::string filename = entry.path().string();
        
        // Skip non-Q-table files
        if (filename.find("qtable_instance_") == std::string::npos) {
            continue;
        }
        
        try {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Unable to open file for loading: " << filename << std::endl;
                continue;
            }
            
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
                    
                    // Merge into the global table
                    if (mergedTable[stateKey].find(actionKey) == mergedTable[stateKey].end()) {
                        mergedTable[stateKey][actionKey] = {qValue, 1};
                    } else {
                        auto& current = mergedTable[stateKey][actionKey];
                        current.first += qValue;
                        current.second += 1;
                    }
                }
            }
            
            file.close();
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing file " << filename << ": " << e.what() << std::endl;
            continue;
        }
    }
    
    // Write the averaged merged Q-table
    std::string mergedFilename = config.sharedDir + "/qtable_merged.bin";
    std::ofstream outFile(mergedFilename, std::ios::binary);
    
    if (!outFile.is_open()) {
        std::cerr << "Unable to open file for saving merged Q-table: " << mergedFilename << std::endl;
        return;
    }
    
    // Save Q-table size
    size_t stateCount = mergedTable.size();
    outFile.write(reinterpret_cast<const char*>(&stateCount), sizeof(size_t));
    
    // Save each state and its actions
    for (const auto& statePair : mergedTable) {
        // Write state key
        size_t stateKeyLength = statePair.first.length();
        outFile.write(reinterpret_cast<const char*>(&stateKeyLength), sizeof(size_t));
        outFile.write(statePair.first.c_str(), stateKeyLength);
        
        // Write number of actions for this state
        size_t actionCount = statePair.second.size();
        outFile.write(reinterpret_cast<const char*>(&actionCount), sizeof(size_t));
        
        // Write each action and its averaged Q-value
        for (const auto& actionPair : statePair.second) {
            // Write action key
            size_t actionKeyLength = actionPair.first.length();
            outFile.write(reinterpret_cast<const char*>(&actionKeyLength), sizeof(size_t));
            outFile.write(actionPair.first.c_str(), actionKeyLength);
            
            // Write averaged Q-value
            float avgQValue = actionPair.second.first / actionPair.second.second;
            outFile.write(reinterpret_cast<const char*>(&avgQValue), sizeof(float));
        }
    }
    
    outFile.close();
    
    if (config.verbose) {
        std::lock_guard<std::mutex> lock(consoleMutex);
        std::cout << "Global merge complete. Merged Q-table written to " << mergedFilename << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Register signal handler
    signal(SIGINT, signalHandler);
    
    // Load configuration
    Config config;
    // TODO: Parse command-line arguments to override config defaults
    
    std::cout << "Parallel RL Coordinator starting up..." << std::endl;
    std::cout << "Using " << config.numInstances << " parallel instances" << std::endl;
    std::cout << "Shared directory: " << config.sharedDir << std::endl;
    
    // Create shared directory if it doesn't exist
    if (!fs::exists(config.sharedDir)) {
        fs::create_directories(config.sharedDir);
        std::cout << "Created shared directory: " << config.sharedDir << std::endl;
    }
    
    // Launch the game instances in separate threads
    std::vector<std::thread> instanceThreads;
    for (int i = 0; i < config.numInstances; i++) {
        instanceThreads.emplace_back(launchGameInstance, i, std::ref(config));
        // Small delay to avoid thundering herd problem
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    // Periodic global merge thread
    auto startTime = std::chrono::steady_clock::now();
    while (isRunning) {
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(
            currentTime - startTime).count();
        
        if (elapsedSeconds >= config.mergeInterval) {
            // Perform global merge
            performGlobalMerge(config);
            startTime = std::chrono::steady_clock::now();
        }
        
        // Sleep a bit to avoid burning CPU
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // Wait for all instance threads to finish
    std::cout << "Waiting for all instances to terminate..." << std::endl;
    for (auto& thread : instanceThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    std::cout << "All instances terminated. Performing final global merge..." << std::endl;
    performGlobalMerge(config);
    
    std::cout << "Parallel RL Coordinator shutdown complete." << std::endl;
    return 0;
}