#ifndef NEURALNET_H
#define NEURALNET_H
#include <vector>
#include <cassert>
#include "neuron.h"

typedef std::vector<neuron> layer;

class neuralNet
{
public:
    neuralNet(const std::vector<unsigned> &layerDef);
    ~neuralNet();
    //feeds data into the neural network and populates the output layer
    void feedForward(const std::vector<double> &inputVals); 
    //adjusts weights and "trains" the neural network
    void backPropigate(const std::vector<double> &targetVals);
    //get results from the output layer
    void getResults(std::vector<double> &resultVals) const;
private:
    std::vector<layer> layers;
    double error;
};

#endif // NEURALNET_H
