#include <iostream>
#include <cmath>
#include "neuralnetwork.hpp"

std::function<double(double)> RELU = [](double x) -> double { if (x < 0) 
                                        return 0;
                                    else 
                                        return x; };
std::function<double(double)> RELUder = [](double x) -> double { if (x < 0) 
                                            return 0;
                                        else 
                                            return 1; };

std::function<double(double)> lin = [](double x) -> double { return x; };
std::function<double(double)> linder = [](double) -> double { return 1; };

std::function<double(double)> logsigmoid = [](double x) -> double { return (1.0 / (1.0 + (std::exp(-x)))); };
std::function<double(double)> logsigmoidder = [=](double x) -> double { return (logsigmoid(x)*(1-logsigmoid(x))); };

std::function<double(double)> unitstep = [](double x) -> double { return x < 0 ? 0.0 : 1.0; };
std::function<double(double)> unitder = [](double) -> double { return 0; };


void xorNN() {
    std::vector<double> v1 = {0,0};
    std::vector<double> v2 = {0,1};
    std::vector<double> v3 = {1,0};
    std::vector<double> v4 = {1,1};
    std::vector<std::vector<double>> testVector = {v1,v2,v3,v4};

    std::vector<double> o1 = {0};
    std::vector<double> o2 = {1};
    std::vector<double> o3 = {1};
    std::vector<double> o4 = {0};
    std::vector<std::vector<double>> outputVector = {o1,o2,o3,o4};

    std::vector<unsigned int> sizeOfLayers = {2,10,1};

    
    NeuralNetwork nn(sizeOfLayers, RELU, RELUder, logsigmoid, logsigmoidder, 
                        std::pair<double,double>(-1,1), crossEntropyBinary);
    nn.train(testVector, outputVector, 4, 0.01);

    for (auto i : testVector) {
        nn.setInput(i);
        std::cout << "Input: " << i[0] << ' ' << i[1] << std::endl; 
        nn.run();
        std::cout << "Output: " << nn.getOutputVector()[0] << std::endl;
    }
}

int main() {
    
    xorNN();
    

    //NeuralNetwork nn(sizeOfLayers, RELU, RELUder, lin, linder);
    //NeuralNetwork nn(sizeOfLayers, unitstep, unitder, unitstep, unitder, std::pair<double,double>(-10,10));

    

}