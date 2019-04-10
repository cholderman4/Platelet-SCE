#ifndef PLATLET_SYSTEM_BUILDER_H_
#define PLATLET_SYSTEM_BUILDER_H_

#include "PlatletSystem.h"

class PlatletSystemBuilder {
public:

    PlatletSystemBuilder(double _epsilon, double _dt);
	~PlatletSystemBuilder();

    Node< thrust::host_vector<bool>, thrust::host_vector<double> > node;
    SpringEdge<thrust::host_vector<double>, thrust::host_vector<unsigned> > springEdge;
    GeneralParams generalParams;

public:

    // Bunch of methods for adding to vectors from XML.
    unsigned addMembraneNode();
    
    unsigned addMembraneEdge();

    // Final goal of SystemBuilder is to create a copy of the system on device.
    std::shared_ptr<PlatletSystem> Create_Platlet_System_On_Device();
};


#endif