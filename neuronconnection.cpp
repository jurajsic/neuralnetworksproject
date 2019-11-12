#include <iostream>
#include "neuron.hpp"

NeuronConnection::NeuronConnection(Neuron *input, double weight, Neuron *output) 
                        : inputNeuron(input), weight(weight), outputNeuron(output)
{
    inputNeuron->addOuptutConnection(this);
    outputNeuron->addInputConnection(this);
}

Neuron* NeuronConnection::getInputNeuron() {
    return inputNeuron;
}

double NeuronConnection::getWeight() {
    return weight;
}

void NeuronConnection::setWeight(double weight) {
    this->weight = weight;
}

Neuron* NeuronConnection::getOutputNeuron() {
    return outputNeuron;
}

void NeuronConnection::computeErrorFunWeightDerAndSave() {
    weightUpdate += outputNeuron->errorFunctionOutputDerivation
                        * outputNeuron->activationFunctionDerivation(outputNeuron->innerPotential)
                        * inputNeuron->getOutput();
    ++weightUpdateCounter;
}

void NeuronConnection::updateWeight(double learningRate, ErrorFunction ef) {
    if (ef == squaredError) {
        // do nothing
    } else if (ef == crossEntropyBinary) {
        weightUpdate = -weightUpdate/weightUpdateCounter;
    } else {
        throw "Not implemented";
    }

    //std::cout << "Updating connection from " << *inputNeuron << " to " << *outputNeuron << " with weight " << weight << " by adding " << weightUpdate << " to it." << std::endl;
    weight += -learningRate * weightUpdate;

    weightUpdate = 0;
    weightUpdateCounter = 0;
}