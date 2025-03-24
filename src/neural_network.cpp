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

void NeuralNetwork::copyWeightsFrom(const NeuralNetwork& other) {
    if (layers.size() != other.layers.size()) {
        throw std::invalid_argument("Networks must have the same number of layers");
    }
    
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& thisLayer = layers[i];
        const auto& otherLayer = other.layers[i];
        
        if (thisLayer.weights.size() != otherLayer.weights.size() ||
            thisLayer.weights[0].size() != otherLayer.weights[0].size()) {
            throw std::invalid_argument("Layer sizes don't match");
        }
        
        // Copy weights
        for (size_t j = 0; j < thisLayer.weights.size(); ++j) {
            for (size_t k = 0; k < thisLayer.weights[j].size(); ++k) {
                thisLayer.weights[j][k] = otherLayer.weights[j][k];
            }
        }
        
        // Copy biases
        for (size_t j = 0; j < thisLayer.biases.size(); ++j) {
            thisLayer.biases[j] = otherLayer.biases[j];
        }
    }
}

void NeuralNetwork::softUpdateFrom(const NeuralNetwork& other, float tau) {
    if (layers.size() != other.layers.size()) {
        throw std::invalid_argument("Networks must have the same number of layers");
    }
    
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& thisLayer = layers[i];
        const auto& otherLayer = other.layers[i];
        
        if (thisLayer.weights.size() != otherLayer.weights.size() ||
            thisLayer.weights[0].size() != otherLayer.weights[0].size()) {
            throw std::invalid_argument("Layer sizes don't match");
        }
        
        // Soft update weights: θ_target = τ*θ_source + (1-τ)*θ_target
        for (size_t j = 0; j < thisLayer.weights.size(); ++j) {
            for (size_t k = 0; k < thisLayer.weights[j].size(); ++k) {
                thisLayer.weights[j][k] = tau * otherLayer.weights[j][k] + 
                                          (1.0f - tau) * thisLayer.weights[j][k];
            }
        }
        
        // Soft update biases
        for (size_t j = 0; j < thisLayer.biases.size(); ++j) {
            thisLayer.biases[j] = tau * otherLayer.biases[j] + 
                                  (1.0f - tau) * thisLayer.biases[j];
        }
    }
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

// DQN Optimizer Implementation

DQNOptimizer::DQNOptimizer(NeuralNetwork& network, float learningRate)
    : network(network), learningRate(learningRate) {
}

void DQNOptimizer::updateWeights(const std::vector<std::vector<float>>& inputs,
                                const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Input and target sizes must match");
    }
    
    auto gradients = calculateGradients(inputs, targets);
    
    // Apply gradients to weights
    auto currentWeights = network.getAllWeights();
    auto currentBiases = network.getAllBiases();
    
    for (size_t i = 0; i < currentWeights.size(); ++i) {
        for (size_t j = 0; j < currentWeights[i].size(); ++j) {
            for (size_t k = 0; k < currentWeights[i][j].size(); ++k) {
                currentWeights[i][j][k] -= learningRate * gradients[i][j][k];
            }
            
            // Update biases
            currentBiases[i][j] -= learningRate * 0.01f; // Simple bias update
        }
    }
    
    network.setAllWeights(currentWeights);
    network.setAllBiases(currentBiases);
}

std::vector<std::vector<std::vector<float>>> DQNOptimizer::calculateGradients(
    const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& targets) {
    
    auto weights = network.getAllWeights();
    std::vector<std::vector<std::vector<float>>> gradients = weights;
    
    // Initialize gradients to zero
    for (auto& layer : gradients) {
        for (auto& neuron : layer) {
            std::fill(neuron.begin(), neuron.end(), 0.0f);
        }
    }
    
    // Simple numerical gradient calculation
    const float epsilon = 0.001f;
    
    for (size_t b = 0; b < inputs.size(); ++b) {
        const auto& input = inputs[b];
        const auto& target = targets[b];
        
        // Calculate current loss
        auto currentOutput = network.forward(input);
        float baseLoss = 0.0f;
        
        for (size_t i = 0; i < currentOutput.size(); ++i) {
            baseLoss += 0.5f * std::pow(currentOutput[i] - target[i], 2);
        }
        
        // Perturb each weight and calculate gradient
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                for (size_t k = 0; k < weights[i][j].size(); ++k) {
                    // Save original weight
                    float originalWeight = weights[i][j][k];
                    
                    // Perturb weight
                    weights[i][j][k] += epsilon;
                    network.setAllWeights(weights);
                    
                    // Calculate perturbed loss
                    auto perturbedOutput = network.forward(input);
                    float perturbedLoss = 0.0f;
                    
                    for (size_t o = 0; o < perturbedOutput.size(); ++o) {
                        perturbedLoss += 0.5f * std::pow(perturbedOutput[o] - target[o], 2);
                    }
                    
                    // Calculate gradient
                    gradients[i][j][k] += (perturbedLoss - baseLoss) / epsilon;
                    
                    // Restore original weight
                    weights[i][j][k] = originalWeight;
                    network.setAllWeights(weights);
                }
            }
        }
    }
    
    // Average gradients over batch
    for (auto& layer : gradients) {
        for (auto& neuron : layer) {
            for (auto& grad : neuron) {
                grad /= inputs.size();
            }
        }
    }
    
    return gradients;
}

} // namespace CarGame