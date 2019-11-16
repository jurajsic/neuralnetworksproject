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

    // this is has to be set before running the neural network
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

    // sets neurons with probability prob as active (dropout)
    void setActiveNeurons(double prob);
public:
    NeuralNetwork() = delete;
    // sizeOfLayers[i] = number of neurons in layer i, 0th layer is input layer, last layer is output layer
    NeuralNetwork(std::vector<unsigned long> sizeOfLayers, 
                  activationFunctionType outputNeuronsActFunType,
                  //std::pair<double,double> weightRange,
                  ErrorFunction ef
                  );
    ~NeuralNetwork();

    // trains the neural network where
    //    trainingVectors - self explanatory
    //    trainingOutput - expected outputs for training vectors
    //    miniBatchSize - the number of vectors that will be processed before weight updates
    //    learninRate - startin learning rate
    //    numOfEpochs - the number of times to go trough whole set of training vectors
    //    weightDecay - self explanatory
    //    sizeOfValidation - the size of randomly selected set of training vectors that are used 
    //                       for validation (right now only outputs the accuracy and error rate 
    //                       of this set after each epoch)
    void train(const std::vector<std::vector<double>> &trainingVectors, 
               const std::vector<std::vector<double>> &trainingOutput,
               unsigned long minibatchSize,
               double learningRate,
               unsigned numOfEpochs,
               double weightDecay = 0.00005,
               unsigned sizeOfValidation = 0);

    // sets the input for neural network
    void setInput(const std::vector<double> &inputVector);
    // runs the network on input
    void run();
    // returns the values of output neurons
    std::vector<double> getOutputVector();
    // computes the error of the set of training vectors which indexes are saved to
    // a vector with starting iterator startIndex and ending iterator endIndex
    // that is a vector: trainingVectors[*startIndex], trainingVectors[*(startIndex + 1)], ..., trainingVectors[*endIndex]
    double computeError(const std::vector<std::vector<double>> &trainingVectors, 
                        const std::vector<std::vector<double>> &expectedOutput,
                        std::vector<unsigned long>::iterator startIndex,
                        std::vector<unsigned long>::iterator endIndex);
    void printOutput();
    void printConnections();
};

#endif