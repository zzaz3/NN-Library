#include "neuralNet.h"

neuralNet::neuralNet(const std::vector<unsigned> &layerDef)
{
    unsigned x = layerDef.size();
    
    //create x amount of layers in the neural net
    for(unsigned layerNum=0;layerNum<x;++layerNum)
    {
        //creates empty layer at layer[layerNum]
        layers.push_back(layer());
        //define number of output connections each neuron will have
        //if its the output layer, the neurons will have 0 output connections
        unsigned numOutputs = layerNum == layerDef.size() - 1 ? 0 : layerDef[layerNum+1];
        //inserts layerDef[layerNum] neurons into the newly created layer, plus a bias neuron 
        for(unsigned neuronNum=0;neuronNum<=layerDef[layerNum];++neuronNum)
        {
            layers.back().push_back(neuron(numOutputs,neuronNum));
        }
        //set bias output to 1
        layers.back().back().setOutputVal(1.0);
        
    }
}

neuralNet::~neuralNet()
{
    
}


void neuralNet::feedForward(const std::vector<double> &inputVals)
{
    //make sure that all input values in the input layer are accounted for 
    assert(inputVals.size()==layers[0].size()-1);
    
    //put input values into the first layer of the neural network
    for(unsigned i=0;i<inputVals.size();++i)
    {
        layers[0][i].setOutputVal(inputVals[i]);
    }
    
    //feed forward
    for(unsigned currentLayer=1;currentLayer<layers.size();++currentLayer)
    {
        layer &prevLayer = layers[currentLayer-1];
        for(unsigned currentNeuron=0; currentNeuron<layers[currentLayer].size()-1;++currentNeuron)
        {
            layers[currentLayer][currentNeuron].feedForward(prevLayer);
        }
    }
    
}

void neuralNet::backPropigate(const std::vector<double> &targetVals)
{
    //calculate net error using RMS formula
    layer &outputLayer = layers.back();
    error = 0.0;
    
    for(unsigned neuronNum=0; neuronNum<outputLayer.size()-1;++neuronNum)
    {
        //find delta - target value minus actual value
        double delta = targetVals[neuronNum]-outputLayer[neuronNum].getOutputVal();
        error += delta*delta;
    }
    //get average error (squared)
    error /= outputLayer.size()-1;
    //get average error
    error = sqrt(error);
    
    //calulate output layer gradients
    for(unsigned neuronNum=0;neuronNum<outputLayer.size()-1;++neuronNum)
    {
        outputLayer[neuronNum].calculateOutputGradients(targetVals[neuronNum]);
    }
    //calculate gradients on hidden layers
    for(unsigned currentLayer=layers.size()-2;currentLayer>0;--currentLayer)
    {
        layer &hiddenLayer = layers[currentLayer];
        layer &nextLayer = layers[currentLayer+1];
        
        for(unsigned neuronNum=0;neuronNum<hiddenLayer.size();++neuronNum)
        {
            hiddenLayer[neuronNum].calculateHiddenGradients(nextLayer);
        }
    }
    //update connection weights for all layers
    for(unsigned layerNum=layers.size()-1;layerNum>0;--layerNum)
    {
        layer &currentLayer = layers[layerNum];
        layer &prevLayer = layers[layerNum-1];
        
        for(unsigned neuronNum=0;neuronNum<currentLayer.size()-1;++neuronNum)
        {
            currentLayer[neuronNum].updateInputWeights(prevLayer);
        }
    }
}

void neuralNet::getResults(std::vector<double> &resultsVals) const
{
    resultsVals.clear();
    //return the results from the output layer
    for(unsigned neuronNum=0;neuronNum<layers.back().size()-1;++neuronNum)
    {
        resultsVals.push_back(layers.back()[neuronNum].getOutputVal());
    }
}
