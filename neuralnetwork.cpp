#include <numeric>
#include <random>
#include <iostream>
#include <algorithm>
#include <iterator>
#include "neuralnetwork.hpp"

NeuralNetwork::NeuralNetwork( 
                  std::vector<unsigned long> sizeOfLayers,
                  activationFunctionType outputNeuronsActFunType, 
                  //std::pair<double,double> weightRange,
                  ErrorFunction ef
                  ) : input(sizeOfLayers[0]), ef(ef), outputNeuronsActivationFunType(outputNeuronsActFunType)
{
    //neurons.reserve(sizeOfLayers.size());

    // create formal neuron for bias
    auto oneFunction = [](double) -> double { return 1; };
    formalNeuron = new Neuron(oneFunction,oneFunction,"formalNeuron");
    formalNeuron->computeOutput();
    //inputLayerNeurons.push_back(formalNeuron);
    
    // create input neurons
    std::vector<Neuron*> inputLayerNeurons;
    //inputLayerNeurons.reserve(sizeOfLayers[0]);
    for (unsigned long i = 0; i != sizeOfLayers[0]; ++i) {
        auto inputActivationFun = [&, i](double) -> double { return input[i]; };
        Neuron *inputNeuron = new Neuron(inputActivationFun, oneFunction, "inputNeuron" + std::to_string(i));
        inputLayerNeurons.push_back(inputNeuron);
    }
    neurons.push_back(inputLayerNeurons);

/*
    // hidden neurons have RELU as activation function
    std::function<double(double)> hiddenActivationFunction = [](double x) -> double { return (x < 0) ? 0.0 : x; };
    std::function<double(double)> hiddenActivationFunctionDerivation = [](double x) -> double { return (x < 0) ? 0.0 : 1.0; };
*/
    // hidden neurons have SELU as activation function
    std::function<double(double)> hiddenActivationFunction = [](double x) -> double { return (x < 0.0) ? (1.05*1.673*(std::exp(x)-1)) : 1.05*x; };
    std::function<double(double)> hiddenActivationFunctionDerivation = [](double x) -> double { return (x < 0.0) ? (1.05*1.673*std::exp(x)) : 1.05; };


    // randomizer for weights
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    //std::uniform_real_distribution<> randomWeight(weightRange.first, weightRange.second);

    // create hidden neurons and connect them with lower layer
    for (std::size_t i = 1; i != sizeOfLayers.size() - 1; ++i) {
        std::vector<Neuron*> newLayer;
        unsigned long numOfNeuronsInLayer = sizeOfLayers[i];
        for (unsigned long j = 0; j != numOfNeuronsInLayer; ++j) {
            Neuron *newNeuron = new Neuron(hiddenActivationFunction, hiddenActivationFunctionDerivation, "hiddenNeuron" + std::to_string(j) + "Layer" + std::to_string(i));
            newLayer.push_back(newNeuron);

            // create connections to the lower layer of neurons
            auto &lowerLayer = neurons.back();
            // weight generated using results from He (2015) designed for RELU
            //std::normal_distribution randomWeight(0.0, 2.0/double(lowerLayer.size()));
            // weight generated using results from LeCun (1990) designed for SELU
            std::normal_distribution<double> randomWeight(0.0, 1.0/double(lowerLayer.size()));
            for (Neuron *lowerLayerNeuron : lowerLayer) {
                NeuronConnection *connection = new NeuronConnection(lowerLayerNeuron, randomWeight(gen), newNeuron);
                connections.push_back(connection);
            }

            // create connection to formal neuron for bias
            NeuronConnection *biasConnection = new NeuronConnection(formalNeuron, 0.0, newNeuron);
            connections.push_back(biasConnection);
        }
        neurons.push_back(newLayer);
    }

    // create output neurons
    unsigned long numOfOutputNeurons = sizeOfLayers.back();
    std::vector<Neuron*> outputLayer;
    for (unsigned long i = 0; i != numOfOutputNeurons; ++i) {
        Neuron *newNeuron = new Neuron("outputNeuron" + std::to_string(i));
        outputLayer.push_back(newNeuron);

        // connect them with lower layer
        auto &lowerLayer = neurons.back();
        // weight generated using results from Glorot & Bengio (2010)
        std::normal_distribution<double> randomWeight(0.0, 2.0/double(lowerLayer.size() + numOfOutputNeurons));
        for (Neuron *lowerLayerNeuron : lowerLayer) {
            NeuronConnection *connection = new NeuronConnection(lowerLayerNeuron, randomWeight(gen), newNeuron);
            connections.push_back(connection);
        }

        // create connection to formal neuron
        NeuronConnection *biasConnection = new NeuronConnection(formalNeuron, 0.0, newNeuron);
        connections.push_back(biasConnection);

    }
    neurons.push_back(outputLayer);

    // process activation functions for output neurons
    std::function<double(double)> outputActivationFunction;
    std::function<double(double)> outputActivationFunctionDerivation;
    switch (outputNeuronsActFunType) {
        case (linear):
            outputActivationFunction = [](double x) -> double { return x; };
            outputActivationFunctionDerivation = [](double) -> double { return 1; };
            break;
        case (logsigmoid):
            outputActivationFunction = [](double x) -> double { return (1.0 / (1.0 + (std::exp(-x)))); };
            outputActivationFunctionDerivation = [=](double x) -> double { return (outputActivationFunction(x)*(1-outputActivationFunction(x))); };
            break;
        case (softmax):
            // assumes that x is not inner potential but exp(innerpotential)
            outputActivationFunction = [&](double x) -> double 
                                        { 
                                            return x / denominatorForSoftmax;
                                        };
            // this "derivation" is set to 1 so computation of partial derivation of error
            // function is easier
            outputActivationFunctionDerivation = [=](double) -> double
                                        {
                                            //double softmaxOrig = outputActivationFunction(x);
                                            //return softmaxOrig * (1 - softmaxOrig);
                                            return 1;
                                        };
            break;
        default:
            throw "Not implemented";
    }

    for (Neuron *outputNeuron : outputLayer) {
        outputNeuron->setActivationFunction(outputActivationFunction);
        outputNeuron->setActivationFunctionDerivation(outputActivationFunctionDerivation);
    }
}

NeuralNetwork::~NeuralNetwork() {
    delete formalNeuron;
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
}

void NeuralNetwork::run() {
    for (auto &layer : neurons) {
        for (Neuron *neuron : layer) {
            neuron->computeInnerPotential();
        }

        // this thing is done so we don't have to recalculate exp in softmax activation function of output neurons
        if (outputNeuronsActivationFunType == softmax) {
            double maxInnerPotential = 0.0;
            for (Neuron *outputNeuron : neurons.back()) {
                double innerPotential = outputNeuron->getInnerPotential();
                maxInnerPotential = (innerPotential > maxInnerPotential) ? innerPotential : maxInnerPotential;                                           
            }
            denominatorForSoftmax = 0.0;
            for (Neuron *outputNeuron : neurons.back()) {
                outputNeuron->computeExpOfInnerPotential(maxInnerPotential);
                denominatorForSoftmax += outputNeuron->getExpOfInnerPotential();
            }
        }

        for (Neuron *neuron : layer) {
            //std::cout << "ID: " << *neuron << std::endl;
            //std::cout << "starting value: " << neuron->getOutput() << std::endl;
            neuron->computeOutput();
            //std::cout << neuron->getOutput() << std::endl;
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
        outputLayer[i]->computeErrorFunctionOutputDerivation(expectedOutput[i], ef);
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
                          unsigned long minibatchSize,
                          double learningRate,
                          unsigned numOfLoops,
                          double weightDecay,
                          unsigned sizeOfValidation) 
{
    // initialize a vector of indexes from 0 to trainingVectors.size()-1
    std::vector<std::size_t> indexes(trainingVectors.size() - sizeOfValidation);
    std::iota(std::begin(indexes), std::end(indexes), 0);
    // stuff for randomization
    std::random_device rd;
    std::mt19937 g(rd());

    // TODO add this whole thing in while block where we decide how long to do this shit
    for(unsigned i=0; i != numOfLoops; ++i) {
        //double error = computeError(trainingVectors, trainingOutput);
        std::cout << "Epoch " << i << std::endl;// " with error " << error << std::endl;
        // shuffle the indexes...
        std::shuffle(indexes.begin(), indexes.end(), g);
        // ... and take first minibatchSize of them for processing
        for (unsigned long j = 0; j * minibatchSize < trainingVectors.size() - sizeOfValidation; ++j) {
            std::cout << "  Batch " << j << std::endl;
            for (unsigned long k = 0; k != minibatchSize && (j*minibatchSize + k) != trainingVectors.size() - sizeOfValidation; ++k) {
                auto index = indexes[j*minibatchSize + k];
                setInput(trainingVectors[index]);
                run();
                //printOutput();
                backpropagate(trainingOutput[index]);
                computeWeightUpdates();
            }
            
            for (NeuronConnection *connection : connections) {
                connection->updateWeight(learningRate, ef, weightDecay);
            }
        }

    
        unsigned right = 0;
        for (auto i = trainingVectors.size() - sizeOfValidation; i < trainingVectors.size(); ++i) {
            setInput(trainingVectors[i]);
            run();
            auto realOutputVec = getOutputVector();
            auto realAns = std::distance(trainingOutput[i].begin(),std::max_element(trainingOutput[i].begin(),trainingOutput[i].end()));
            auto expecAns = std::distance(realOutputVec.begin(),std::max_element(realOutputVec.begin(),realOutputVec.end()));
            if (realAns == expecAns)
                ++right;
        }
        std::cout << double(right) / double(sizeOfValidation) << std::endl;
    }
}

void NeuralNetwork::printOutput() {
    for (double o : getOutputVector()) {
        std::cout << o << ' ';
    }
    std::cout << std::endl;
}

double NeuralNetwork::computeError(const std::vector<std::vector<double>> &trainingVectors, 
                                   const std::vector<std::vector<double>> &expectedOutput)
{
    double error = 0;
    for (std::size_t i = 0; i != trainingVectors.size(); ++i) {
        double sizeOfSet = trainingVectors.size();
        setInput(trainingVectors[i]);
        run();
        auto output = getOutputVector();
        for (std::size_t j = 0; j != expectedOutput[i].size(); ++j) {
            double realOutput = output[j];
            double expectOutput = expectedOutput[i][j];
            if (ef == meanSquaredError) {
                double temp = realOutput - expectOutput;
                error +=  temp * temp / 2.0;
                error = error / sizeOfSet;
            } else if (ef == crossEntropyBinary) {
                error -= (expectOutput*log(realOutput) + (1-expectOutput)*log(1-realOutput)) / sizeOfSet;
            } else {
                throw "not implemented";
            }
        }
    }
    return error;
}

void NeuralNetwork::printConnections() {
    for (NeuronConnection *c : connections) {
        std::cout << *c << std::endl;
    }
}