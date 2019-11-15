#include <cmath>
#include <iostream>
#include "neuron.hpp"

Neuron::Neuron(std::string id) : id(id) {}

Neuron::Neuron(const std::function<double(double)> &activationFunction, 
               const std::function<double(double)> &activationFunctionDerivation,
               std::string id) : id(id)
{
    setActivationFunction(activationFunction);
    setActivationFunctionDerivation(activationFunctionDerivation);
}

void Neuron::addInputConnection(NeuronConnection *inputConnection) {
    if (inputConnection->getOutputNeuron() != this) {
        throw "Wrong connection";
    }
    inputConnections.push_back(inputConnection);
}

void Neuron::addOuptutConnection(NeuronConnection *outputConnection) {
    if (outputConnection->getInputNeuron() != this) {
        throw "Wrong connection";
    }
    outputConnections.push_back(outputConnection);
}

double Neuron::getOutput() {
    return output;
}

void Neuron::computeInnerPotential() {
    innerPotential = 0;
    for (NeuronConnection *inputConnection : inputConnections) {
        innerPotential += inputConnection->getWeight() * inputConnection->getInputNeuron()->getOutput();
    }
}

double Neuron::getInnerPotential() {
    return innerPotential;
}

void Neuron::computeOutput() {
    // this is true only if we have softmax
    if (expOfInnerPotential != -1.0) {
        output = activationFunction(expOfInnerPotential);
    } else {
        output = activationFunction(innerPotential);
    }
}

void Neuron::setActivationFunction(const std::function<double(double)> &activationFunction) {
    this->activationFunction = activationFunction;
}

void Neuron::setActivationFunctionDerivation(const std::function<double(double)> &activationFunctionDerivation) {
    this->activationFunctionDerivation = activationFunctionDerivation;
}

void Neuron::computeErrorFunctionOutputDerivation(double expectedOutput, ErrorFunction ef) {
    if (outputConnections.size() == 0) {
        if (ef == meanSquaredError) {
            errorFunctionOutputDerivation = output - expectedOutput;
        } else if (ef == crossEntropyBinary) {
            errorFunctionOutputDerivation = expectedOutput/output - (1-expectedOutput)/(1-output);
        } else if (ef == crossEntropy) {
            // this is different than basic back propagation, it assumes that
            // "derivation" of activation function is set to constant 1
            errorFunctionOutputDerivation = expectedOutput - output; 
        } else {
            throw "Not implemented";
        }
    } else {
        errorFunctionOutputDerivation = 0;
        for (NeuronConnection *outputConnection : outputConnections) {
            Neuron *outputN = outputConnection->getOutputNeuron();
            double errorFunDer = outputN->errorFunctionOutputDerivation;
            double activFunDer = outputN->activationFunctionDerivation(outputN->innerPotential);
            errorFunctionOutputDerivation += errorFunDer * activFunDer * outputConnection->getWeight();
        }
    }
}

// computing of exp uses idea from
// https://stats.stackexchange.com/questions/304758/softmax-overflow 
void Neuron::computeExpOfInnerPotential(double maxOutputLayerInnerPotential) {
    expOfInnerPotential = std::exp(innerPotential - maxOutputLayerInnerPotential);
}
double Neuron::getExpOfInnerPotential() {
    return expOfInnerPotential;
}

std::ostream &operator<<(std::ostream &os, Neuron const &n) { 
    return os << n.id;
}