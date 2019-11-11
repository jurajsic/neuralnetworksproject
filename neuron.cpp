#include "neuron.hpp"

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
    /*if (!needToRecomputeInnerPotential) {
        return;
    }*/

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

void Neuron::computeOutput() {
    computeInnerPotential();
    output = activationFunction(innerPotential);
    /*for (Neuron *outputNeuron : outputNeurons) {
        outputNeuron->needToRecomputeInnerPotential = true;
    } */
}

void Neuron::setActivationFunction(const std::function<double(double)> &activationFunction) {
    this->activationFunction = activationFunction;
}

void Neuron::setActivationFunctionDerivation(const std::function<double(double)> &activationFunctionDerivation) {
    this->activationFunctionDerivation = activationFunctionDerivation;
}

void Neuron::computeErrorFunctionOutputDerivation(double expectedOutput) {
    if (outputConnections.size() == 0) {
        errorFunctionOutputDerivation = output - expectedOutput;
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

std::ostream &operator<<(std::ostream &os, Neuron const &n) { 
    return os << n.id;
}