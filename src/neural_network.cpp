#include "neural_network.h"
#include <iostream>
#include <cmath>

namespace CarGame {

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes) {
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("Network must have at least 2 layers");
    }
    
    // Create hidden layers with ReLU activation
    for (size_t i = 0; i < layerSizes.size() - 2; ++i) {
        layers.emplace_back(layerSizes[i], layerSizes[i + 1], relu);
    }
    
    // Create output layer with linear activation
    layers.emplace_back(layerSizes[layerSizes.size() - 2], 
                        layerSizes[layerSizes.size() - 1], 
                        linear);
    
    // Initialize weights
    initializeWeights();
}

void NeuralNetwork::initializeWeights(unsigned int seed) {
    std::mt19937 rng(seed);
    
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];
        int inputSize = layer.weights[0].size();
        int outputSize = layer.weights.size();
        
        // Xavier initialization
        float scale = std::sqrt(6.0f / (inputSize + outputSize));
        std::uniform_real_distribution<float> dist(-scale, scale);
        
        // Initialize weights
        for (auto& neuron : layer.weights) {
            for (auto& weight : neuron) {
                weight = dist(rng);
            }
        }
        
        // Initialize biases to small values
        for (auto& bias : layer.biases) {
            bias = dist(rng) * 0.1f;
        }
    }
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) const {
    std::vector<float> activations = input;
    
    for (const auto& layer : layers) {
        std::vector<float> layerOutputs(layer.biases.size());
        
        for (size_t i = 0; i < layer.weights.size(); ++i) {
            float sum = layer.biases[i];
            
            for (size_t j = 0; j < layer.weights[i].size(); ++j) {
                sum += layer.weights[i][j] * activations[j];
            }
            
            layerOutputs[i] = layer.activation(sum);
        }
        
        activations = layerOutputs;
    }
    
    return activations;
}

std::vector<std::vector<std::vector<float>>> NeuralNetwork::getAllWeights() const {
    std::vector<std::vector<std::vector<float>>> allWeights;
    
    for (const auto& layer : layers) {
        allWeights.push_back(layer.weights);
    }
    
    return allWeights;
}

std::vector<std::vector<float>> NeuralNetwork::getAllBiases() const {
    std::vector<std::vector<float>> allBiases;
    
    for (const auto& layer : layers) {
        allBiases.push_back(layer.biases);
    }
    
    return allBiases;
}

void NeuralNetwork::setAllWeights(const std::vector<std::vector<std::vector<float>>>& weights) {
    if (weights.size() != layers.size()) {
        throw std::invalid_argument("Weight array size doesn't match number of layers");
    }
    
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& thisLayer = layers[i];
        const auto& layerWeights = weights[i];
        
        if (thisLayer.weights.size() != layerWeights.size() ||
            (layerWeights.size() > 0 && thisLayer.weights[0].size() != layerWeights[0].size())) {
            throw std::invalid_argument("Layer weight dimensions don't match");
        }
        
        thisLayer.weights = layerWeights;
    }
}

void NeuralNetwork::setAllBiases(const std::vector<std::vector<float>>& biases) {
    if (biases.size() != layers.size()) {
        throw std::invalid_argument("Bias array size doesn't match number of layers");
    }
    
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& thisLayer = layers[i];
        const auto& layerBiases = biases[i];
        
        if (thisLayer.biases.size() != layerBiases.size()) {
            throw std::invalid_argument("Layer bias dimensions don't match");
        }
        
        thisLayer.biases = layerBiases;
    }
}

std::vector<int> NeuralNetwork::getLayerSizes() const {
    std::vector<int> sizes;
    
    if (!layers.empty()) {
        // Input layer size
        sizes.push_back(layers[0].weights[0].size());
        
        // Hidden and output layers
        for (const auto& layer : layers) {
            sizes.push_back(layer.weights.size());
        }
    }
    
    return sizes;
}

} // namespace CarGame