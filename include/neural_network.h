#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <functional>
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace CarGame {

class NeuralNetwork {
public:
    struct Layer {
        std::vector<std::vector<float>> weights;
        std::vector<float> biases;
        std::function<float(float)> activation;
        
        Layer(int inputSize, int outputSize, 
              std::function<float(float)> activation)
            : weights(outputSize, std::vector<float>(inputSize, 0.0f)),
              biases(outputSize, 0.0f),
              activation(activation) {
        }
    };
    
    NeuralNetwork(const std::vector<int>& layerSizes);
    
    // Initialize network with Xavier initialization
    void initializeWeights(unsigned int seed = 42);
    
    // Forward pass through the network
    std::vector<float> forward(const std::vector<float>& input) const;
    
    // Copy weights from another network (for target network updates)
    void copyWeightsFrom(const NeuralNetwork& other);
    
    // Soft update weights (for target network)
    void softUpdateFrom(const NeuralNetwork& other, float tau);
    
    // Getters for weights and biases (for saving/loading)
    std::vector<std::vector<std::vector<float>>> getAllWeights() const;
    std::vector<std::vector<float>> getAllBiases() const;
    
    // Setters for weights and biases (for loading)
    void setAllWeights(const std::vector<std::vector<std::vector<float>>>& weights);
    void setAllBiases(const std::vector<std::vector<float>>& biases);
    
    // Get layer sizes
    std::vector<int> getLayerSizes() const;
    
private:
    std::vector<Layer> layers;
    
    // Activation functions
    static float relu(float x) { return x > 0.0f ? x : 0.0f; }
    static float linear(float x) { return x; }
};

class DQNOptimizer {
public:
    DQNOptimizer(NeuralNetwork& network, float learningRate = 0.001f);
    
    // Update network weights using MSE loss
    void updateWeights(const std::vector<std::vector<float>>& inputs,
                      const std::vector<std::vector<float>>& targets);
    
private:
    NeuralNetwork& network;
    float learningRate;
    
    // Numerical gradient calculation
    std::vector<std::vector<std::vector<float>>> calculateGradients(
        const std::vector<std::vector<float>>& inputs,
        const std::vector<std::vector<float>>& targets);
};

} // namespace CarGame