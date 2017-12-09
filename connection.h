#ifndef CONNECTION_H
#define CONNECTION_H
#include <cstdlib>
//class that holds the weights for every connection a neuron has to the next layer
//upon creation, each weight is assigned a small, random value (between 0 and 1)
class connection
{
public:
    connection();
    ~connection();
    double getWeight(){return weight;}
    double getChangeInWeight(){return changeInWeight;}
    void setWeight(double x){weight=x;}
    void setChangeInWeight(double x){changeInWeight=x;}
private:
    double weight;
    double changeInWeight;
};

#endif // CONNECTION_H
