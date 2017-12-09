#include "trainingData.h"

trainingData::trainingData(const char* num)
{
    m_trainingDataFile.open(num,std::ios::binary);
    trainingSet=*num;
    
}

trainingData::~trainingData()
{
    
}

unsigned trainingData::getNextInputs(std::vector<double> &inputVals)
{
    inputVals.clear();

    char ch;
    
    while(!m_trainingDataFile.eof())
    {
        for(int i=0;i<(28*28);++i)
        {
            m_trainingDataFile.get(ch);
            double temp = std::abs((double)ch);
            inputVals.push_back(temp);
        }
        break;
    }

    return inputVals.size();
}

unsigned trainingData::getTargetOutputs(std::vector<double> &targetOutputVals)
{
    targetOutputVals.clear();
    if(trainingSet=='0')
        targetOutputVals.push_back(.00);
    else if(trainingSet=='1')
        targetOutputVals.push_back(.1);
    else if(trainingSet=='2')
        targetOutputVals.push_back(.2);
    else if(trainingSet=='3')
        targetOutputVals.push_back(.3);
    else if(trainingSet=='4')
        targetOutputVals.push_back(.4);
    else if(trainingSet=='5')
        targetOutputVals.push_back(.5);

    return targetOutputVals.size();
}

