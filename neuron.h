#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include <cmath>
#include "connection.h"

class neuron
{
public:
    typedef std::vector<neuron> layer;
    neuron(unsigned numOutputs, unsigned newIndex);
    ~neuron();
    void setOutputVal(double val) {outputValue = val;}
    double getOutputVal() const {return outputValue;}
    //takes all the ouput values from the previous layer and feeds into this neuron
    void feedForward(layer &prevLayer);
    //finds slope of the error for neurons in the output layer
    void calculateOutputGradients(double targetVal);
    //finds slope of the error for neurons in the hidden layers
    void calculateHiddenGradients(const layer &nextLayer);
    //updates the weights for the neuron that are stored in the previous layer
    void updateInputWeights(layer &prevLayer);
private:
    double learningRate;
    double momentum;
    static double sigmoidFunction(double x);
    static double sigmoidFunctionDerivative(double x);
    double sumDOW(const layer &nextLayer);
    double outputValue;
    unsigned my_index;
    double gradient;
    std::vector<connection> outputWeights;
};

#endif // NEURON_H
