#include <iostream>
#include "neuralnetwork.hpp"

int main() {
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

    std::vector<unsigned int> sizeOfLayers = {2,2,2,1};

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

    NeuralNetwork nn(sizeOfLayers, RELU, RELUder, lin, linder);
    nn.train(testVector, outputVector, 4, 0.1);
    nn.setInput({0,1});
    nn.run();
    std::cout << nn.getOutputVector()[0] << std::endl;
}