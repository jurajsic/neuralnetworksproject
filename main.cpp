#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "neuralnetwork.hpp"

void xorNN() {
    std::vector<double> v1 = {0,0};
    std::vector<double> v2 = {0,1};
    std::vector<double> v3 = {1,0};
    std::vector<double> v4 = {1,1};
    std::vector<std::vector<double>> testVector = {v1,v2,v3,v4};

    std::vector<double> o1 = {1,0};
    std::vector<double> o2 = {0,1};
    std::vector<double> o3 = {0,1};
    std::vector<double> o4 = {1,0};
    std::vector<std::vector<double>> outputVector = {o1,o2,o3,o4};

    std::vector<unsigned long> sizeOfLayers = {2,10,2};
  
    //NeuralNetwork nn(sizeOfLayers, logsigmoid, std::pair<double,double>(-1,1), crossEntropyBinary);
    NeuralNetwork nn(sizeOfLayers, linear, std::pair<double,double>(-5,5), meanSquaredError);
    //NeuralNetwork nn(sizeOfLayers, softmax, std::pair<double,double>(-5,5), crossEntropy);
    nn.train(testVector, outputVector, 4, 0.01, 10000);

    for (auto i : testVector) {
        nn.setInput(i);
        std::cout << "Input: " << i[0] << ' ' << i[1] << std::endl; 
        nn.run();
        nn.printOutput();
    }
    //nn.printConnections();
}

std::vector<std::vector<double>> readVectorFile(std::string fileName) {
    std::ifstream file(fileName);
    std::vector<std::vector<double>> content;
    if (file.is_open()) {
        std::string line;
        while(std::getline(file, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::stringstream divider(line);
            double cell;
            std::vector<double> row;
            while (divider >> cell) {
                row.push_back(cell);
            }
            content.push_back(row);
        }
    } else {
        throw "Could not read the file!";
    }
    return content;
}

std::vector<unsigned> readLabelFile(std::string fileName) {
    std::ifstream file(fileName);
    std::vector<unsigned> labels;
    if (file.is_open()) {
        std::string line;
        while(std::getline(file, line)) {
            std::stringstream ss(line);
            unsigned label;
            ss >> label;
            labels.push_back(label);
        }
    } else {
        throw "Could not read the file!";
    }
    return labels;
}

int main(int argc, char **argv) {
    /*auto content = readCSVFile("data.test");
    for (auto &r : content) {
        for (auto &c : r) {
            std::cout << c << ',';
        }
        std::cout << std::endl;
    }
    return 0;*/

    if (argc < 3)
    {
        xorNN();
        return 0;
    }

    auto vectors = readVectorFile("data/MNIST_DATA/mnist_train_vectors.csv");
    auto labels = readLabelFile("data/MNIST_DATA/mnist_train_labels.csv");

    std::cout << "Processing files" << std::endl;

    unsigned maxLabel = 0;
    for (unsigned label : labels) {
        if (label > maxLabel) {
            maxLabel = label;
        }
    }
    
    std::vector<std::vector<double>> outputs;
    for (unsigned label : labels) {
        std::vector<double> output(maxLabel + 1, 0.0);
        output[label] = 1.0;
        outputs.push_back(output);
    }

    std::vector<unsigned long> sizeOfLayers = {vectors[0].size(), 100, maxLabel + 1};

    std::cout << "Creating neural network" << std::endl;
  
    //NeuralNetwork nn(sizeOfLayers, softmax, std::pair<double,double>(-1,1), crossEntropy);
    NeuralNetwork nn(sizeOfLayers, linear, std::pair<double,double>(-1,1), meanSquaredError);
    nn.train(vectors, outputs, 1000, 0.01, 1);
 
    vectors = readVectorFile("data/MNIST_DATA/mnist_test_vectors.csv");
    std::ofstream outFile("testLabels");
    for (auto &v : vectors) {
        nn.setInput(v);
        nn.run();
        auto res = nn.getOutputVector();
        unsigned maxindex = 0;
        double max = res[0];
        for (std::size_t i = 1; i != maxLabel; ++i) {
            if (res[i] > max) {
                max = res[i];
                maxindex = i;
            }
        }
        outFile << maxindex << std::endl;
    }
 }