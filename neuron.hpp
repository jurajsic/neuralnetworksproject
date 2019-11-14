#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <functional>

enum ErrorFunction { meanSquaredError, crossEntropy, crossEntropyBinary };

class Neuron;

class NeuronConnection {
private:
    Neuron *inputNeuron;
    double weight;
    Neuron *outputNeuron;

    double weightUpdate = 0;
    unsigned weightUpdateCounter = 0;
    
    friend std::ostream& operator<<(std::ostream&, NeuronConnection const&);
public:
    NeuronConnection() = delete;
    NeuronConnection(Neuron *input, double weight, Neuron *output);
    Neuron* getInputNeuron();
    double getWeight();
    //void setWeight(double weight);
    Neuron* getOutputNeuron();

    void computeErrorFunWeightDerAndSave();
    void updateWeight(double learningRate, ErrorFunction ef, double weightDecay = 0.0);
};

class Neuron {
private:
    friend class NeuronConnection;
    // id of neuron used for debuging
    std::string id;
    friend std::ostream& operator<<(std::ostream&, Neuron const&);

    std::vector<NeuronConnection*> inputConnections;
    std::vector<NeuronConnection*> outputConnections;

    double innerPotential = 0.0;

    double output = 0.0;
    std::function<double(double)> activationFunction;
    std::function<double(double)> activationFunctionDerivation;

    /** backpropagation stuff **/
    double errorFunctionOutputDerivation = 0.0;
    
    // this is only used if activation function is softmax, so we do not recompute exp multiple times
    // it is not actually exp of inner potential but exp(innerPotential - maxOutputLayerInnerPotential)
    // where maxOutputLayerInnerPotential is maximal inner potential of neuron in output layer
    // so we do not have problems with too large numbers
    double expOfInnerPotential = -1.0;

public:
    Neuron(std::string id);
    Neuron(const std::function<double(double)> &activationFunction, 
           const std::function<double(double)> &activationFunctionDerivation,
           std::string id);

    void addInputConnection(NeuronConnection *inputConnection);
    void addOuptutConnection(NeuronConnection *outputConnection);
    
    void computeInnerPotential();
    double getInnerPotential();
    void computeOutput();
    double getOutput();
    
    void setActivationFunction(const std::function<double(double)> &activationFunction);
    void setActivationFunctionDerivation(const std::function<double(double)> &activationFunctionDerivation);

    // for backpropagation
    void computeErrorFunctionOutputDerivation(double expectedOutput = 0, ErrorFunction ef = meanSquaredError);
    //void computeErrorFunctionWeightsDerivation(); // should add to errorFunctionWeightsDerivation not replace
    //void updateWeights(double learningRate); // should zero errorFunctionWeightsDerivation
    //double getErrorFunctionDerivation();

    void computeExpOfInnerPotential(double maxOutputLayerInnerPotential);
    double getExpOfInnerPotential();
};

#endif