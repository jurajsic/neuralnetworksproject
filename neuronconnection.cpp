#include <iostream>
#include <iomanip>
#include <cmath>
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

Neuron* NeuronConnection::getOutputNeuron() {
    return outputNeuron;
}

void NeuronConnection::computeErrorFunWeightDerAndSave() {
    weightUpdate += outputNeuron->errorFunctionOutputDerivation
                        * outputNeuron->activationFunctionDerivation(outputNeuron->innerPotential)
                        * inputNeuron->getOutput();
    ++weightUpdateCounter;
}

void NeuronConnection::updateWeight(double learningRate, ErrorFunction ef, double weightDecay) {
    if (ef == meanSquaredError) {
        weightUpdate = weightUpdate/weightUpdateCounter;
    } else if (ef == crossEntropyBinary || ef == crossEntropy) {
        weightUpdate = -weightUpdate/weightUpdateCounter;
    } else {
        throw "Not implemented";
    }

    //rmsdrop
    double smoothingTerm = 0.9;
    learningRateAdaptation = smoothingTerm*learningRateAdaptation + (1-smoothingTerm)*weightUpdate*weightUpdate;

    // SGD computing learning rate
    //weight += -learningRate * weightUpdate;

    // rmsdrop computing learning rate
    weight += -learningRate * weightUpdate / std::sqrt(learningRateAdaptation + 1e-8);

    // weight decay
    weight = (1 - weightDecay) * weight;
    
    weightUpdate = 0;
    weightUpdateCounter = 0;
}

std::ostream& operator<<(std::ostream& os, NeuronConnection const& c) {
    return os << *c.inputNeuron << "    " << std::setprecision(5) << c.weight << "    " << *c.outputNeuron;
}