#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <algorithm>
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
  
    
    auto t_start = std::chrono::high_resolution_clock::now();

    //NeuralNetwork nn(sizeOfLayers, logsigmoid, std::pair<double,double>(-1,1), crossEntropyBinary);
    //NeuralNetwork nn(sizeOfLayers, linear, std::pair<double,double>(-5,5), meanSquaredError);
    NeuralNetwork nn(sizeOfLayers, softmax, crossEntropy);
    nn.train(testVector, outputVector, 4, 0.01, 1000, 0.00005);

    
    auto t_end = std::chrono::high_resolution_clock::now();

    double elapsed_time_m = std::chrono::duration<double, std::ratio<60>>(t_end-t_start).count();
    std::cout << "Time taken to train: " << elapsed_time_m << std::endl;

    for (auto i : testVector) {
        nn.setInput(i);
        std::cout << "Input: " << i[0] << ' ' << i[1] << std::endl; 
        nn.run();
        nn.printOutput();
    }
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
                // input is divided by 255 to get values from 0 to 1
                row.push_back(cell/255.0);
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

void runNetworkAndWriteOutput(std::vector<std::vector<double>> &inputVectors, std::string outFileName, NeuralNetwork &nn) {
    std::ofstream outFile(outFileName);
    for (auto &v : inputVectors) {
        nn.setInput(v);
        nn.run();
        auto res = nn.getOutputVector();
        unsigned label = std::distance(res.begin(),std::max_element(res.begin(),res.end()));
        outFile << label << std::endl;
    }
}

int main(int argc, char **argv) {

    auto t_start = std::chrono::high_resolution_clock::now();

    if (argc < 6)
    {
        xorNN();
        return 0;
    }

    auto vectors = readVectorFile("../data/mnist_train_vectors.csv");
    auto labels = readLabelFile("../data/mnist_train_labels.csv");

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

    std::vector<unsigned long> sizeOfLayers;
    sizeOfLayers.push_back(vectors[0].size());
    for (int i = 1; i < argc-4; ++i)
        sizeOfLayers.push_back(std::stoul(argv[1]));
    sizeOfLayers.push_back(maxLabel + 1);

    std::cout << "Creating neural network" << std::endl;

    double learnRate = std::stod(argv[argc-3]);
    NeuralNetwork nn(sizeOfLayers, softmax, crossEntropy);
    //NeuralNetwork nn(sizeOfLayers, linear, std::pair<double,double>(-1,1), meanSquaredError);
    nn.train(vectors, outputs, 32, learnRate, std::stoul(argv[argc-2]), 0.00005, 0);

    // run neural network on training data
    runNetworkAndWriteOutput(vectors, argv[argc-2], nn);

    // run neural network on test data
    vectors = readVectorFile("../data/mnist_test_vectors.csv");
    runNetworkAndWriteOutput(vectors, argv[argc-1], nn);

    auto t_end = std::chrono::high_resolution_clock::now();

    double elapsed_time_m = std::chrono::duration<double, std::ratio<60>>(t_end-t_start).count();
    std::cout << "Time taken for the whole tango: " << elapsed_time_m << " minutes" << std::endl;
 }