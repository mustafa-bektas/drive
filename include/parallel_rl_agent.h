#pragma once

#include "rl_agent.h"
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <chrono>

namespace CarGame {

class ParallelRLAgent : public RLAgent {
public:
    ParallelRLAgent(int instanceId, const std::string& sharedDirectory, 
                   float learningRate = 0.9f, float discountFactor = 0.95f, 
                   float explorationRate = 1.0f);
    ~ParallelRLAgent();
    
    // Override training method to include parallel operations
    void trainFromReplay(int batchSize = 32) override;
    
    // Methods for parallel operation
    void startSharingThread();
    void stopSharingThread();
    
    // Q-table sharing
    void exportQTable(const std::string& filename);
    void importQTable(const std::string& filename);
    void mergeQTable(const std::map<std::string, std::map<std::string, float>>& otherQTable);
    void checkAndLoadSharedQTables();
    
private:
    int instanceId;
    std::string sharedDirectory;
    std::thread sharingThread;
    std::atomic<bool> isRunning;
    std::mutex qTableMutex;
    
    // Thread function for periodic sharing
    void sharingThreadFunction();
};

} // namespace CarGame