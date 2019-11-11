#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <functional>

class Neuron;

class NeuronConnection {
private:
    Neuron *inputNeuron;
    double weight;
    Neuron *outputNeuron;

    double weightUpdate = 0;
public:
    NeuronConnection() = delete;
    NeuronConnection(Neuron *input, double weight, Neuron *output);
    Neuron* getInputNeuron();
    double getWeight();
    void setWeight(double weight);
    Neuron* getOutputNeuron();

    void computeErrorFunWeightDerAndSave();
    void updateWeight(double learningRate);
};

class Neuron {
private:
    friend class NeuronConnection;

    double bias = 0;
    // input neurons with their weights
   /* std::vector<Neuron*> inputNeurons; // TODO decide if keeping as pair or two vectors
    std::vector<double> weights;
    std::vector<Neuron*> outputNeurons;*/

    std::vector<NeuronConnection*> inputConnections;
    std::vector<NeuronConnection*> outputConnections;

    double output = 0;
    std::function<double(double)> activationFunction;
    std::function<double(double)> activationFunctionDerivation;

    /** backpropagation stuff **/
    double errorFunctionOutputDerivation = 0;
    // weight update for each weight
    //std::vector<double> errorFunctionWeightsDerivation;

    double innerPotential = 0;
    //bool needToRecomputeInnerPotential = true; // TODO do I need this????
    void computeInnerPotential();
public:
    Neuron(const std::function<double(double)> &activationFunction, 
           const std::function<double(double)> &activationFunctionDerivation);

    void addInputConnection(NeuronConnection *inputConnection);
    void addOuptutConnection(NeuronConnection *outputConnection);

    double getOutput();
    void computeOutput();
    
    void setActivationFunction(const std::function<double(double)> &activationFunction);
    void setActivationFunctionDerivation(const std::function<double(double)> &activationFunctionDerivation);

    // for backpropagation
    void computeErrorFunctionOutputDerivation(double expectedOutput = 0);
    //void computeErrorFunctionWeightsDerivation(); // should add to errorFunctionWeightsDerivation not replace
    //void updateWeights(double learningRate); // should zero errorFunctionWeightsDerivation
    //double getErrorFunctionDerivation();
};

#endif