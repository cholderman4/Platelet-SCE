#ifndef KINETIC_ENERGY_H_
#define KINETIC_ENERGY_H_


#include <vector>


#include "ICalculateParameter.h"


class NodeData;


class KineticEnergy : public ICalculateParameter {    
    private:

    NodeData& nodeData;

    double energy{ 0.0 };

    public:

    KineticEnergy(NodeData& _nodeData);

    void calculate();

    double getParameter();
};

#endif