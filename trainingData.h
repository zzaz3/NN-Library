#ifndef TRAININGDATA_H
#define TRAININGDATA_H
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

class trainingData
{
public:
    trainingData(const char* num);
    ~trainingData();
    bool isEof() { return m_trainingDataFile.eof(); }
    
    // gets the input values from the file to be feed into the networks input layer
    unsigned getNextInputs(std::vector<double> &inputVals);
    //get the expected output values
    unsigned getTargetOutputs(std::vector<double> &targetOutputVals);

private:
    std::ifstream m_trainingDataFile;
    char trainingSet;
};

#endif // TRAININGDATA_H
