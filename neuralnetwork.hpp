#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "neuron.hpp"

class NeuralNetwork {
private:
    /** all neurons **/
    //std::vector<Neuron> inputNeurons;
    //std::vector<Neuron> outputNeurons;
    // hiddenNeurons[i][j] = neuron j in ith hidden layer
    // neurons[0] = input layer
    // neurons[last] = output layer
    std::vector<std::vector<Neuron*>> neurons; // TODO maybe have here also input and output layers
    std::vector<NeuronConnection*> connections;
    std::vector<double> input;

    ErrorFunction ef = squaredError;

    /** backpropagation stuff **/
    /*std::vector<std::vector<double>> outputNeuronsDerivatives;
    // derivatives[i][j] = partial derivative of error function with respect to neuron j in layer i 
    std::vector<std::vector<double>> hiddenNeuronsDerivatives; /**/
    
    void backpropagate(const std::vector<double> &expectedOutput);
    void computeWeightUpdates();
    void printOutput();
public:
    NeuralNetwork() = delete;
    NeuralNetwork(//unsigned int numOfHiddenLayers, 
                  std::vector<unsigned int> sizeOfLayers, 
                  std::function<double(double)> &hiddenActivationFunction,
                  std::function<double(double)> &hiddenActivationFunctionDerivation,
                  std::function<double(double)> &outputActivationFunction,
                  std::function<double(double)> &outputActivationFunctionDerivation,
                  std::pair<double,double> weightRange,
                  ErrorFunction ef
                  );
    ~NeuralNetwork();
    // TODO constructors setting parameters + add parameters?
    void train(const std::vector<std::vector<double>> &trainingVectors, 
               const std::vector<std::vector<double>> &trainingOutput,
               unsigned int minibatchSize,
               double learningRate);
    void setInput(const std::vector<double> &inputVector);
    void run();
    std::vector<double> getOutputVector();
    double computeError(const std::vector<std::vector<double>> &trainingVectors, 
                        const std::vector<std::vector<double>> &expectedOutput);
};

#endif