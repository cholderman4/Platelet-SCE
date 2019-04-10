#ifndef PLATLET_SYSTEM_H_
#define PLATLET_SYSTEM_H_

// Should hold all the data structures needed for the platlet model.
// All class methods should be the default methods (forward declarations??) with more useful versions in the .cu file.

#include "SystemStructures.h"

class PlatletStorage;

template<typename vec_D, typename vec_U>
struct SpringEdge {
    /* Vector size is 2*numSpringEdges.
    For now, each node is connected to 2 spring edges.
    Each spring corresponds to two consecutive vector entries, 
    corresponding to its two connected neighbors. */

    vec_U nodeConnections;

    vec_D len_0;
};

template<typename vec_B, typename vec_D>
struct Node {
    // Holds a set of xyz coordinates for a single node.
    vec_D pos_x;
    vec_D pos_y;
    vec_D pos_z;
    
    vec_D velocity;

    /* These are apparently not needed. Only the 
    magnitude of velocity to know when equilibrium 
    is reached.
    
    vec_D vel_x;
    vec_D vel_y;
    vec_D vel_z; */

    vec_D force_x;
    vec_D force_y;
    vec_D force_z;

    vec_B isFixed;


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
    Node< thrust::device_vector<bool>, thrust::device_vector<double> > node;
    SpringEdge<thrust::device_vector<double>, thrust::device_vector<unsigned> > springEdge;
    GeneralParams generalParams;

    std::shared_ptr<PlatletStorage> pltStorage;

public:

    PlatletSystem();

    void PlatletSystem::assignPltStorage(std::shared_ptr<PlatletStorage> _pltStorage);


    void initializePltSystem(unsigned N, unsigned E);


    void solvePltSystem();


    void solvePltForces();


    void setPltNodes();


    void setPltSpringEdge();


    void printPoints();


    void printConnections();
    

    void printForces();

};


#endif