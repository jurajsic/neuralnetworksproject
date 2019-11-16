#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <random>
#include "neuron.hpp"

enum activationFunctionType { linear, logsigmoid, softmax };

class NeuralNetwork {
private:
    /* Vector of all neurons
     * hiddenNeurons[i][j] = neuron j in ith hidden layer
     * neurons[0] = input layer
     * neurons[last] = output layer
     */
    std::vector<std::vector<Neuron*>> neurons;
    // formal neuron used for computing bias
    Neuron* formalNeuron;
    // vector of all connections between neurons
    std::vector<NeuronConnection*> connections;

    std::vector<double> input;

    ErrorFunction ef;
    activationFunctionType outputNeuronsActivationFunType;

    // used if output layer has softmax as activation function
    double denominatorForSoftmax;
    void computeDenominatorForSoftmax();

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::default_random_engine gen;
    
    void backpropagate(const std::vector<double> &expectedOutput);
    void computeWeightUpdates();
    void setActiveNeurons(double prob);
public:
    NeuralNetwork() = delete;
    NeuralNetwork(std::vector<unsigned long> sizeOfLayers, 
                  activationFunctionType outputNeuronsActFunType,
                  //std::pair<double,double> weightRange,
                  ErrorFunction ef
                  );
    ~NeuralNetwork();
    void train(const std::vector<std::vector<double>> &trainingVectors, 
               const std::vector<std::vector<double>> &trainingOutput,
               unsigned long minibatchSize,
               double learningRate,
               unsigned numOfLoops,
               double weightDecay,
               unsigned sizeOfValidation = 0);
    void setInput(const std::vector<double> &inputVector);
    void run();
    std::vector<double> getOutputVector();
    double computeError(const std::vector<std::vector<double>> &trainingVectors, 
                        const std::vector<std::vector<double>> &expectedOutput);
    void printOutput();
    void printConnections();
};

#endif