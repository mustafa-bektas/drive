#pragma once

#include <vector>
#include <functional>
#include <random>
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
    
    // create network with specified layers
    NeuralNetwork(const std::vector<int>& layerSizes);
    
    // forward pass (inference)
    std::vector<float> forward(const std::vector<float>& input) const;
    
    // getters/setters for weights and biases
    std::vector<std::vector<std::vector<float>>> getAllWeights() const;
    std::vector<std::vector<float>> getAllBiases() const;
    void setAllWeights(const std::vector<std::vector<std::vector<float>>>& weights);
    void setAllBiases(const std::vector<std::vector<float>>& biases);
    
    // get sizes
    std::vector<int> getLayerSizes() const;
    
private:
    std::vector<Layer> layers;
    
    // activation funcs
    static float relu(float x) { return x > 0.0f ? x : 0.0f; }
    static float linear(float x) { return x; }
    
    // xavier init
    void initializeWeights(unsigned int seed = 42);
};

} // namespace CarGame