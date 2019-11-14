#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
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
    NeuralNetwork nn(sizeOfLayers, softmax, std::pair<double,double>(-5,5), crossEntropy);
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
/*
    for (double i = 0.0; i <= 1.0; i += 0.1) {
        for (double j = 0.0; j <= 1.0; j += 0.1) {
            std::cout << "Input: " << i << ' ' << j << std::endl;
            nn.setInput(std::vector<double>{i,j});
            nn.run();
            nn.printOutput();
        }
    }*/
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

    if (argc < 4)
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

    std::vector<unsigned long> sizeOfLayers = {vectors[0].size(), 600, maxLabel + 1};

    std::cout << "Creating neural network" << std::endl;
  
    auto t_start = std::chrono::high_resolution_clock::now();

    double maxWeight = std::stod(argv[1]);
    double learnRate = std::stod(argv[2]);
    NeuralNetwork nn(sizeOfLayers, softmax, std::pair<double,double>(-maxWeight,maxWeight), crossEntropy);
    //NeuralNetwork nn(sizeOfLayers, linear, std::pair<double,double>(-1,1), meanSquaredError);
    nn.train(vectors, outputs, 32, learnRate, 1, 0.00005);

    auto t_end = std::chrono::high_resolution_clock::now();

    double elapsed_time_m = std::chrono::duration<double, std::ratio<60>>(t_end-t_start).count();
    std::cout << "Time taken to train: " << elapsed_time_m << std::endl;

 /*
    std::vector<double> inp = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,18,18,18,126,136,175,26,166,255,247,127,0,0,0,0,0,0,0,0,0,0,0,0,30,36,94,154,170,253,253,253,253,253,225,172,253,242,195,64,0,0,0,0,0,0,0,0,0,0,0,49,238,253,253,253,253,253,253,253,253,251,93,82,82,56,39,0,0,0,0,0,0,0,0,0,0,0,0,18,219,253,253,253,253,253,198,182,247,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,156,107,253,253,205,11,0,43,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,1,154,253,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,190,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,190,253,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,241,225,160,108,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,240,253,253,119,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,186,253,253,150,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,93,252,253,187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,249,253,249,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,130,183,253,253,207,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,148,229,253,253,253,250,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,114,221,253,253,253,253,201,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,66,213,253,253,253,253,198,81,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,171,219,253,253,253,253,195,80,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,172,226,253,253,253,253,244,133,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,136,253,253,253,212,135,132,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    nn.setInput(inp);
    nn.run();
    nn.printOutput();
*/
 
    vectors = readVectorFile("data/MNIST_DATA/mnist_test_vectors.csv");
    std::ofstream outFile(argv[3]);
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