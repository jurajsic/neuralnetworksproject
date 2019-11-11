#include <numeric>
#include <random>
#include <iostream>
#include "neuralnetwork.hpp"

/*
TODO
    - initial weigths - constant or based on some logic
    - learning rate - constant or add variable based on current iteration
    - mean sqared error ???
*/

NeuralNetwork::NeuralNetwork(//unsigned int numOfHiddenLayers, 
                  std::vector<unsigned int> sizeOfLayers, 
                  std::function<double(double)> &hiddenActivationFunction,
                  std::function<double(double)> &hiddenActivationFunctionDerivation,
                  std::function<double(double)> &outputActivationFunction,
                  std::function<double(double)> &outputActivationFunctionDerivation
                  // TODO default weight
                  // TODO learning rate
                  ) : input(sizeOfLayers[0])
{
    neurons.reserve(sizeOfLayers.size());
    // create input neurons
    std::vector<Neuron*> inputLayerNeurons;
    auto oneFunction = [](double) -> double { return 1; };
    Neuron *formalNeuron = new Neuron(oneFunction,oneFunction,"formalNeuron"); // formal input neuron for bias
    inputLayerNeurons.push_back(formalNeuron); // formal input neuron for bias
    
    for (unsigned int i = 0; i != sizeOfLayers[0]; ++i) {
        auto inputActivationFun = [&, i](double) -> double { return input[i]; };
        Neuron *inputNeuron = new Neuron(inputActivationFun, oneFunction, "inputNeuron" + std::to_string(i));
        inputLayerNeurons.push_back(inputNeuron);
    }
    neurons.push_back(inputLayerNeurons);

    // create hidden + output neurons and connect them with lower layer
    for (std::size_t i = 1; i != sizeOfLayers.size(); ++i) {
        std::vector<Neuron*> newLayer;
        for (unsigned int j = 0; j != sizeOfLayers[i]; ++j) {
            Neuron *newNeuron;
            if (i == (sizeOfLayers.size()-1)) {
                newNeuron = new Neuron(outputActivationFunction, outputActivationFunctionDerivation, "outputNeuron" + std::to_string(j));
            } else {
                newNeuron = new Neuron(hiddenActivationFunction, hiddenActivationFunctionDerivation, "hiddenNeuron" + std::to_string(j) + "Layer" + std::to_string(i));
            }
            newLayer.push_back(newNeuron);

            for (Neuron *lowerLayerNeuron : neurons[i-1]) {
                NeuronConnection *connection = new NeuronConnection(lowerLayerNeuron, 0.1, newNeuron); // TODO default weight
                connections.push_back(connection);
            }

            // create connection to also fromal neuron for bias (but not for the first hidden layer because it was already done)
            if (i != 1) {
                NeuronConnection *biasConnection = new NeuronConnection(formalNeuron, 0.1, newNeuron); // TODO default weight
                connections.push_back(biasConnection);
            }
        }
        neurons.push_back(newLayer);
    }
}

NeuralNetwork::~NeuralNetwork() {
    for (auto &layer : neurons) {
        for (Neuron *neuronToDelete : layer) {
            delete neuronToDelete;
        }
    }
    for (NeuronConnection *connectionToDelete : connections) {
        delete connectionToDelete;
    }
}

void NeuralNetwork::setInput(const std::vector<double> &inputVector) {
    for (std::size_t i = 0; i != input.size(); ++i) {
        input[i] = inputVector[i];
    }
    /*
    auto &inputNeurons = neurons[0];
    for (std::size_t i = 0; i != inputNeurons.size(); ++i) {
        // TODO change to setting some input vector of doubles as member variable which is referenced in the lambdas of input neurons
        double input = inputVector[i];
        inputNeurons[i].setActivationFunction([=](double) -> double { return input; });
        inputNeurons[i].computeOutput();
    }
    */
}

void NeuralNetwork::run() {
    for (auto &layer : neurons) {
        for (Neuron *neuron : layer) {
            std::cout << "ID: " << *neuron << std::endl;
            std::cout << neuron->getOutput() << std::endl;
            neuron->computeOutput();
            std::cout << neuron->getOutput() << std::endl;
        }
    }
}

std::vector<double> NeuralNetwork::getOutputVector() {
    std::vector<double> outputVector;
    auto &outputNeurons = neurons.back();
    for (Neuron *outputNeuron : outputNeurons) {
        outputVector.push_back(outputNeuron->getOutput());
    }
    return outputVector;
}

void NeuralNetwork::backpropagate(const std::vector<double> &expectedOutput) {
    // initialize derivatives for output layer
    auto outputLayer = neurons.back();
    for (std::size_t i = 0; i != expectedOutput.size(); ++i) {
        outputLayer[i]->computeErrorFunctionOutputDerivation(expectedOutput[i]);
    }

    // backpropagate for hidden layers
    for (auto layer = neurons.rbegin() + 1; layer != (neurons.rend() - 1); ++layer) {
        for (Neuron *hiddenNeuron : *layer) {
            hiddenNeuron->computeErrorFunctionOutputDerivation();
        }
    }
}

void NeuralNetwork::computeWeightUpdates() {
    for (NeuronConnection *connection : connections) {
        connection->computeErrorFunWeightDerAndSave();
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>> &trainingVectors, 
                          const std::vector<std::vector<double>> &trainingOutput,
                          unsigned int minibatchSize,
                          double learningRate) 
{
    // initialize a vector of indexes from 0 to trainingVectors.size()-1
    std::vector<std::size_t> indexes(trainingVectors.size());
    std::iota(std::begin(indexes), std::end(indexes), 0);
    // stuff for randomization
    std::random_device rd;
    std::mt19937 g(rd());

    // TODO add this whole thing in while block where we decide how long to do this shit
    for(int i=0; i != 1000; ++i) {
        std::cout << "Iteration: " << i << std::endl;
        // shuffle the indexes...
        std::shuffle(indexes.begin(), indexes.end(), g);
        // ... and take first minibatchSize of them for processing
        for (std::size_t i = 0; i != minibatchSize; ++i) {
            auto index = indexes[i];
            setInput(trainingVectors[index]);
            run();
            backpropagate(trainingOutput[index]);
            computeWeightUpdates();
        }

        for (NeuronConnection *connection : connections) {
            connection->updateWeight(learningRate);
        }
    }
}