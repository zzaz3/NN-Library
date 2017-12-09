#include <iostream>
#include <vector>
#include "neuralNet.h"
#include "trainingData.h"



void showVectorVals(std::string label, std::vector<double> &v)
{
	std::cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		std::cout << v[i] << "";
        if(v.size()>100 && i%28==0)
            std::cout << std::endl;
	}
	std::cout << std::endl;
}

void isCorrect(std::vector<double> &targetVals,std::vector<double> &resultVals)
{
    if(resultVals.back()>=(targetVals.back()-0.1) && resultVals.back()<=(targetVals.back()+0.1))
        std::cout << "Correct!";
    else 
        std::cout << "Incorrect!";
}


int main()
{
    //create training data object for a handwritten "5"
    trainingData trainData("5");
    
    //create neural network object with 784 input nodes and 1 output node (784-10-1)
    std::vector<unsigned> layerDef;
    layerDef.push_back(784);
    layerDef.push_back(10);
    layerDef.push_back(1);
    neuralNet handWriting(layerDef);
    
    
    std::vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    
    //train the neural network until it reaches the end of the training data
    while (!trainData.isEof())
    {
        ++trainingPass;
        std::cout << std::endl << "Pass " << trainingPass << std::endl;

        //get input data and feed it forward into the neural network
        if (trainData.getNextInputs(inputVals) != layerDef[0]) 
        {
            break;
        }
        showVectorVals(" Inputs:", inputVals);
        handWriting.feedForward(inputVals);

        //get the results from the output layer of the neural network
        handWriting.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        //train the network through back propigation
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == layerDef.back());
        handWriting.backPropigate(targetVals);

        //check to see if the neural network was roughly correct
        isCorrect(targetVals,resultVals);
    }

    std::cout << std::endl << "Done" << std::endl;
}