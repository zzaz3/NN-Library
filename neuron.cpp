#include "neuron.h"

neuron::neuron(unsigned numOutputs, unsigned newIndex)
{
    //numOutputs is the number of connections the neuron has to neurons in the next layer
    //create connection objects and store them in the neurons' outputWeights vector
    for(unsigned c=0;c<numOutputs;++c)
    {
        outputWeights.push_back(connection());
    }
    
    my_index=newIndex;
    
    //define learning rate and momentum
    learningRate=0.15;
    momentum=0.5;
}

neuron::~neuron()
{
    
}

void neuron::updateInputWeights(layer &prevLayer)
{
    //goes back through every neuron in each previous layer and updates the weights
    for(unsigned neuronNum=0;neuronNum<prevLayer.size();++neuronNum)
    {
        neuron &currentNeuron = prevLayer[neuronNum];
        //calculate old change in weights
        double oldCIW = currentNeuron.outputWeights[my_index].getChangeInWeight();
        //calulate new change in weights based on momentum,learning rate, and the gradient
        double newCIW = 
            learningRate
            * currentNeuron.getOutputVal()
            * gradient
            + momentum
            * oldCIW;
        
        //updates values of the weights in the neuron's connection
        currentNeuron.outputWeights[my_index].setChangeInWeight(newCIW);
        currentNeuron.outputWeights[my_index].setWeight
        (currentNeuron.outputWeights[my_index].getWeight()+newCIW);
    }
}

double neuron::sumDOW(const layer &nextLayer)
{
    double sum = 0.0;
    
    //find the sum of the derivative of the weights in the next layer
    //used to calculate the gradients for the neurons in the hidden layers
    for(unsigned neuronNum=0;neuronNum<nextLayer.size()-1;++neuronNum)
    {
        sum += outputWeights[neuronNum].getWeight() * nextLayer[neuronNum].gradient;
    }
    return sum;
}

void neuron::calculateHiddenGradients(const layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    gradient = dow * neuron::sigmoidFunctionDerivative(outputValue);
}

void neuron::calculateOutputGradients(double targetVal)
{
    double delta = targetVal-outputValue;
    gradient=delta * neuron::sigmoidFunctionDerivative(outputValue);
}

double neuron::sigmoidFunction(double x)
{
    return tanh(x);
}

double neuron::sigmoidFunctionDerivative(double x)
{
    return  1- x*x;
}

void neuron::feedForward(layer &prevLayer)
{
    double sum = 0.0;
    
    //sum the previous layer's outputs and bias node from previous layer
    for(unsigned neuronNum=0;neuronNum<prevLayer.size();++neuronNum)
    {
        sum+=prevLayer[neuronNum].getOutputVal() *
                prevLayer[neuronNum].outputWeights[neuronNum].getWeight();
    }
    
    outputValue = neuron::sigmoidFunction(sum);
}
