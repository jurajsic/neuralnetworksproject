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
}

void NeuronConnection::updateWeight(double learningRate) {
    weight += -learningRate * weightUpdate;
    weightUpdate = 0;
}