#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "neuron.hpp"

enum activationFunctionType { linear, logsigmoid, softmax };

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

    ErrorFunction ef;
    activationFunctionType outputNeuronsActivationFunType;

    // used if output layer has softmax as activation function
    double *denominatorForSoftmax = nullptr;
    void computeDenominatorForSoftmax();

    /** backpropagation stuff **/
    /*std::vector<std::vector<double>> outputNeuronsDerivatives;
    // derivatives[i][j] = partial derivative of error function with respect to neuron j in layer i 
    std::vector<std::vector<double>> hiddenNeuronsDerivatives; /**/
    
    void backpropagate(const std::vector<double> &expectedOutput);
    void computeWeightUpdates();
public:
    NeuralNetwork() = delete;
    NeuralNetwork(//unsigned int numOfHiddenLayers, 
                  std::vector<unsigned long> sizeOfLayers, 
                  //activationFunctionType hiddenNeuronsActFunType,
                  activationFunctionType outputNeuronsActFunType,  /*
                  std::function<double(double)> &hiddenActivationFunction,
                  std::function<double(double)> &hiddenActivationFunctionDerivation,
                  std::function<double(double)> &outputActivationFunction,
                  std::function<double(double)> &outputActivationFunctionDerivation, */
                  std::pair<double,double> weightRange,
                  ErrorFunction ef
                  );
    ~NeuralNetwork();
    // TODO constructors setting parameters + add parameters?
    void train(const std::vector<std::vector<double>> &trainingVectors, 
               const std::vector<std::vector<double>> &trainingOutput,
               unsigned long minibatchSize,
               double learningRate,
               unsigned numOfLoops,
               double weightDecay);
    void setInput(const std::vector<double> &inputVector);
    void run();
    std::vector<double> getOutputVector();
    double computeError(const std::vector<std::vector<double>> &trainingVectors, 
                        const std::vector<std::vector<double>> &expectedOutput);
    void printOutput();
    void printConnections();
};

#endif