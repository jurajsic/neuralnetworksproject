#include <iostream>
#include <iomanip>
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
    if (ef == meanSquaredError) {
        weightUpdate = weightUpdate/weightUpdateCounter;
    } else if (ef == crossEntropyBinary || ef == crossEntropy) {
        weightUpdate = -weightUpdate/weightUpdateCounter;
    } else {
        throw "Not implemented";
    }

    //std::cout << "Updating connection from " << *inputNeuron << " to " << *outputNeuron << " with weight " << weight << " by adding " << weightUpdate << " to it." << std::endl;
    weight += -learningRate * weightUpdate;
    outputNeuron->needToRecomputeInnerPotential = true;

    weightUpdate = 0;
    weightUpdateCounter = 0;
}

std::ostream& operator<<(std::ostream& os, NeuronConnection const& c) {
    return os << *c.inputNeuron << "    " << std::setprecision(5) << c.weight << "    " << *c.outputNeuron;
}