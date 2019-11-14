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
    // TODO raise expection if inputConnection->outputNeuron != this
    inputConnections.push_back(inputConnection);
}

void Neuron::addOuptutConnection(NeuronConnection *outputConnection) {
    // TODO raise expection if outputConnection->inputNeuron != this
    outputConnections.push_back(outputConnection);
}

double Neuron::getOutput() {
    return output;
}

void Neuron::computeInnerPotential() {
    //if (!needToRecomputeInnerPotential) {
    //    return;
    //}

    //innerPotential = bias;
    /*for (std::size_t i = 0; i != inputNeurons.size(); ++i) {
        innerPotential += weights[i] * inputNeurons[i]->getOutput();
    }*/

    innerPotential = 0;
    for (NeuronConnection *inputConnection : inputConnections) {
        innerPotential += inputConnection->getWeight() * inputConnection->getInputNeuron()->getOutput();
    }

    //needToRecomputeInnerPotential = false;
}

double Neuron::getInnerPotential() {
    //computeInnerPotential();
    return innerPotential;
}

void Neuron::computeOutput() {
    //computeInnerPotential();
    //double oldOutput = output;
    // this is true only if we have softmax
    if (expOfInnerPotential != -1.0) {
        std::cout << "asdaw" << std::endl;
        output = activationFunction(expOfInnerPotential);
    } else {
        output = activationFunction(innerPotential);
    }
    /*if (oldOutput != output) {
        for (NeuronConnection *outputConnection : outputConnections) {
            outputConnection->getOutputNeuron()->needToRecomputeInnerPotential = true;
        } 
    }*/
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
            //errorFunctionOutputDerivation = expectedOutput/output;
            // this is different than basic back propagation, it assumes that "derivation" of activation function
            // is set to constant 1
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