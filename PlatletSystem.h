#ifndef PLATLET_SYSTEM_H_
#define PLATLET_SYSTEM_H_

// Should hold all the data structures needed for the platlet model.
// All class methods should be the default methods (forward declarations??) with more useful versions in the .cu file.

#include "SystemStructures.h"


struct SpringEdge {
    // Store ID of each node connected to edge.
    thrust::device_vector<unsigned> node_L;
    thrust::device_vector<unsigned> node_R;
};


struct Node {
    // Holds a set of xyz coordinates for a single node.
    thrust::device_vector<double> pos_x;
    thrust::device_vector<double> pos_y;
    thrust::device_vector<double> pos_z;
    
    thrust::device_vector<double> vel_x;
    thrust::device_vector<double> vel_y;
    thrust::device_vector<double> vel_z;

    thrust::device_vector<double> force_x;
    thrust::device_vector<double> force_y;
    thrust::device_vector<double> force_z;
};


class PlatletSystem {
public:
    Node node;
    SpringEdge springEdge;

public:

    PlatletSystem();


    void initializePltSystem(unsigned N);


    void solvePltSystem();


    void solvePltForces();


    void setPltNodes(unsigned N);


    void setPltSpringEdge(unsigned N);


    void printPoints();


    void printConnections();
    

    void printForces();

};


#endif