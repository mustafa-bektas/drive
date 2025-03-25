#pragma once

#include <vector>
#include <functional>
#include <random>
#include <stdexcept>

namespace CarGame {

/**
 * Simple neural network for inference of trained models
 * Simplified to only include functionality needed for model inference
 */
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
    
    // Constructor creates a network with the specified layer sizes
    NeuralNetwork(const std::vector<int>& layerSizes);
    
    // Forward pass through the network (inference)
    std::vector<float> forward(const std::vector<float>& input) const;
    
    // Getters and setters for weights and biases (for loading from file)
    std::vector<std::vector<std::vector<float>>> getAllWeights() const;
    std::vector<std::vector<float>> getAllBiases() const;
    void setAllWeights(const std::vector<std::vector<std::vector<float>>>& weights);
    void setAllBiases(const std::vector<std::vector<float>>& biases);
    
    // Get layer sizes
    std::vector<int> getLayerSizes() const;
    
private:
    std::vector<Layer> layers;
    
    // Activation functions
    static float relu(float x) { return x > 0.0f ? x : 0.0f; }
    static float linear(float x) { return x; }
    
    // Initialize weights with Xavier initialization
    void initializeWeights(unsigned int seed = 42);
};

} // namespace CarGame