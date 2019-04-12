#ifndef PLATLET_SYSTEM_H_
#define PLATLET_SYSTEM_H_

// Should hold all the data structures needed for the platlet model.
// All class methods should be the default methods (forward declarations??) with more useful versions in the .cu file.

#include "SystemStructures.h"

// class PlatletStorage;

struct SpringEdge {
    /* Vector size is 2*numSpringEdges.
    For now, each node is connected to 2 spring edges.
    Each spring corresponds to two consecutive vector entries, 
    corresponding to its two connected neighbors. */

    thrust::device_vector<unsigned> nodeConnections;

    thrust::device_vector<double> len_0;
};

struct Node {
    // Holds a set of xyz coordinates for a single node.
    thrust::device_vector<double> pos_x;
    thrust::device_vector<double> pos_y;
    thrust::device_vector<double> pos_z;
    
    thrust::device_vector<double> velocity;

    thrust::device_vector<double> force_x;
    thrust::device_vector<double> force_y;
    thrust::device_vector<double> force_z;

    thrust::device_vector<bool> isFixed;


    /* Keep track of the number of springs connected to each node
    Possibly not needed if we just resize to memNodeCount * maxConnectedSpringCount
    and fill empty connections with ULONGMAX */
    //thrust::device_vector<unsigned> numConnectedSprings;
};


struct GeneralParams {

    /* For tracking the simulation while it is running. */
    bool runSim = true;
    bool currentTime = 0.0;
    unsigned iterationCounter = 0;


    /* Parameters related to nodes, edges, and connections. */ 
    unsigned springEdgeCount;
    unsigned memNodeCount;
    unsigned maxConnectedSpringCount = 2;


    /* Parameters used in various formulae. */
    double dt = 0.1;
	double viscousDamp = 3.769911184308; //???
	double temperature = 300.0;
	double kB = 1.3806488e-8;
	double memNodeMass = 1.0;
};


class PlatletSystem {
public:
    Node node;
    SpringEdge springEdge;
    GeneralParams generalParams;

    //std::shared_ptr<PlatletStorage> pltStorage;

public:

    PlatletSystem();

    //void PlatletSystem::assignPltStorage(std::shared_ptr<PlatletStorage> _pltStorage);


    void initializePltSystem();


    void solvePltSystem();


    void solvePltForces();


    void setPltNodes();


    void setPltSpringEdge();


    void printPoints();


    void printConnections();
    

    void printForces();

};


#endif