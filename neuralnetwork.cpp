#include "neuralnetwork.hpp"

/*
TODO
    - initial weigths - constant or based on some logic
    - learning rate - constant or add variable based on current iteration
    - minibatch - implement random choice of training vectors in each iteration + what size
    - mean sqared error ???
*/

NeuralNetwork::NeuralNetwork(//unsigned int numOfHiddenLayers, 
                  std::vector<unsigned int> sizeOfLayers, 
                  std::function<double(double)> &activationFunction,
                  std::function<double(double)> &activationFunctionDerivation
                  // TODO default weigth
                  // TODO learning rate
                  ) : input(sizeOfLayers[0])
{
    // create input neurons
    std::vector<Neuron> inputLayerNeurons;
    auto zeroFunction = [](double) -> double { return 0; };
    for (unsigned int i = 0; i != sizeOfLayers[0]; ++i) {
        auto inputActivationFun = [&, i](double) -> double { return input[i]; };
        inputLayerNeurons.push_back(Neuron(inputActivationFun, zeroFunction));
    }
    neurons.push_back(inputLayerNeurons);

    // create hidden + output neurons and connect them with lower layer
    for (std::size_t i = 1; i != sizeOfLayers.size(); ++i) {
        auto newLayer = std::vector<Neuron>(sizeOfLayers[i], Neuron(activationFunction, activationFunctionDerivation));
        for (Neuron &newNeuron : newLayer) {
            for (Neuron &lowerLayerNeuron : neurons[i-1]) {
                NeuronConnection connection(&lowerLayerNeuron, 0, &newNeuron); // TODO default weight?????
                connections.push_back(connection);
            }
        }
    }
}

void NeuralNetwork::setInput(const std::vector<double> &inputVector) {
    input = inputVector;
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
        for (auto &neuron : layer) {
            neuron.computeOutput();
        }
    }
}

std::vector<double> NeuralNetwork::getOutputVector() {
    std::vector<double> outputVector;
    auto &outputNeurons = neurons.back();
    for (auto &outputNeuron : outputNeurons) {
        outputVector.push_back(outputNeuron.getOutput());
    }
    return outputVector;
}

void NeuralNetwork::backpropagate(const std::vector<double> &expectedOutput) {
    // initialize derivatives for output layer
    auto outputLayer = neurons.back();
    for (std::size_t i = 0; i != expectedOutput.size(); ++i) {
        outputLayer[i].computeErrorFunctionOutputDerivation(expectedOutput[i]);
    }
    
    // backpropagate for hidden layers
    for (auto layer = neurons.rbegin() + 1; layer != (neurons.rend() - 1); ++layer) {
        for (auto &neuron : *layer) {
            neuron.computeErrorFunctionOutputDerivation();
        }
    }
}

void NeuralNetwork::computeWeightUpdates() {
    for (NeuronConnection &connection : connections) {
        connection.computeErrorFunWeightDerAndSave();
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>> &trainingVectors, 
                          const std::vector<std::vector<double>> &trainingOutput) {
    // TODO add this whole thing in while block where we decide how long to do this shit
    std::vector<std::size_t> trainingExamplesIndices; // TODO get some random
    for (auto index : trainingExamplesIndices) {
        setInput(trainingVectors[index]);
        run();
        backpropagate(trainingOutput[index]);
        computeWeightUpdates();
    }

    double learningRate; // TODO decide what learning rate
    for (NeuronConnection &connection : connections) {
        connection.updateWeight(learningRate);
    }
}