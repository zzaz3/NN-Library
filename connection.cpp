#include "connection.h"

connection::connection()
{
    weight = rand() / double(RAND_MAX);
    changeInWeight = rand() / double(RAND_MAX);
}
connection::~connection()
{
    
}
