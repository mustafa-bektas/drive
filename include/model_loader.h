#pragma once

#include "neural_network.h"
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

namespace CarGame {

class ModelLoader {
public:
    // load model from python export
    static bool loadModelFromPython(const std::string& filename, NeuralNetwork& network) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "can't open file: " << filename << std::endl;
            return false;
        }
        
        // get architecture
        std::string line;
        if (!std::getline(file, line) || line != "network_architecture") {
            std::cerr << "bad file format - missing arch section" << std::endl;
            return false;
        }
        
        // read layer sizes
        std::vector<int> layerSizes;
        while (std::getline(file, line)) {
            if (line == "weights") break;
            try {
                layerSizes.push_back(std::stoi(line));
            } catch (const std::exception& e) {
                std::cerr << "parse error on layer size: " << e.what() << std::endl;
                return false;
            }
        }
        
        // need at least in/out layers
        if (layerSizes.size() < 2) {
            std::cerr << "need at least in/out layers" << std::endl;
            return false;
        }
        
        // recreate network with right architecture
        NeuralNetwork newNetwork(layerSizes);
        
        // read weights section
        std::vector<std::vector<std::vector<float>>> weights;
        std::vector<std::vector<float>> currentLayerWeights;
        int currentLayer = 0;
        int currentNeuron = 0;
        
        // init weights structure
        for (size_t i = 1; i < layerSizes.size(); i++) {
            int outputSize = layerSizes[i];
            int inputSize = layerSizes[i-1];
            
            std::vector<std::vector<float>> layerWeights;
            for (int j = 0; j < outputSize; j++) {
                layerWeights.push_back(std::vector<float>(inputSize, 0.0f));
            }
            weights.push_back(layerWeights);
        }
        
        // read weights
        while (std::getline(file, line)) {
            if (line == "biases") break;
            
            std::istringstream iss(line);
            float weight;
            std::vector<float> neuronWeights;
            
            while (iss >> weight) {
                neuronWeights.push_back(weight);
            }
            
            // store weights
            if (!neuronWeights.empty()) {
                if (currentNeuron < weights[currentLayer].size() && 
                    neuronWeights.size() == weights[currentLayer][currentNeuron].size()) {
                    weights[currentLayer][currentNeuron] = neuronWeights;
                } else {
                    std::cerr << "weight dims don't match" << std::endl;
                    return false;
                }
                
                currentNeuron++;
                if (currentNeuron >= weights[currentLayer].size()) {
                    currentNeuron = 0;
                    currentLayer++;
                }
            }
        }
        
        // read biases
        std::vector<std::vector<float>> biases;
        for (size_t i = 1; i < layerSizes.size(); i++) {
            biases.push_back(std::vector<float>(layerSizes[i], 0.0f));
        }
        
        currentLayer = 0;
        while (std::getline(file, line) && currentLayer < biases.size()) {
            std::istringstream iss(line);
            float bias;
            std::vector<float> layerBiases;
            
            while (iss >> bias) {
                layerBiases.push_back(bias);
            }
            
            if (layerBiases.size() == biases[currentLayer].size()) {
                biases[currentLayer] = layerBiases;
                currentLayer++;
            } else {
                std::cerr << "bias dims don't match" << std::endl;
                return false;
            }
        }
        
        // set weights and biases
        newNetwork.setAllWeights(weights);
        newNetwork.setAllBiases(biases);
        
        // replace the input network
        network = newNetwork;
        
        std::cout << "loaded model: " << filename << std::endl;
        return true;
    }
};

} // namespace CarGame