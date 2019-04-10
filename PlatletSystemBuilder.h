#ifndef PLATLET_SYSTEM_BUILDER_H_
#define PLATLET_SYSTEM_BUILDER_H_

#include "PlatletSystem.h"

class PlatletSystemBuilder {
public:
    Node< thrust::host_vector<bool>, thrust::host_vector<double> > node;
    SpringEdge<thrust::host_vector<double>, thrust::host_vector<unsigned> > springEdge;

public:

    // Bunch of methods for adding to vectors from XML.

    std::shared_ptr<PlatletSystem> create();
};


#endif