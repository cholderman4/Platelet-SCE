#ifndef PLATLET_SYSTEM_H_
#define PLATLET_SYSTEM_H_

// Should hold all the data structures needed for the platlet model.
// All class methods should be the default methods (forward declarations??) with more useful versions in the .cu file.

#include "SystemStructures.h"

class PlatletStorage;

struct SpringEdge {   
    thrust::device_vector<unsigned> nodeID_L;
    thrust::device_vector<unsigned> nodeID_R;

    thrust::device_vector<double> len_0;

    double stiffness{ 30.0 };
    unsigned count{ 0 };

    // ****************************************
    // These will be used if we calculate force by spring
    // instead of by node, then sort, reduce (by key).
    /* thrust::device_vector<unsigned> tempNodeID;
    thrust::device_vector<double> tempForce_x;
    thrust::device_vector<double> tempForce_y;
    thrust::device_vector<double> tempForce_z;

    thrust::device_vector<unsigned> reducedNodeID;
    // Probably don't need these. Can just += to force vectors.
    thrust::device_vector<double> reducedForce_x;
    thrust::device_vector<double> reducedForce_y;
    thrust::device_vector<double> reducedForce_z; */
    // ****************************************
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

    double mass{ 1.0 };
    unsigned count{ 0 };

};

struct MembraneNode : public Node {

    thrust::device_vector<bool> isFixed;

    // Vector size is M * N, 
    // where    M = maxConnectedSpringCount
    // and      N = memNodeCount.
    // Each entry corresponds to the ID of a spring that node is connected to.
    thrust::device_vector<unsigned> connectedSpringID;
    thrust::device_vector<unsigned> connectedSpringCount;

    // Used for indexing purposes.
    unsigned maxConnectedSpringCount{ 2 };
};


/* struct InternalNode : public Node {

}; */

struct SimulationParams {

    // For tracking the simulation while it is running. 
    bool runSim{ true };
    double currentTime{ 0.0 };
    unsigned iterationCounter{ 0 };
    unsigned maxIterations{ 10000 };
    unsigned printFileStepSize{ 200 };

};


struct GeneralParams {

    // Parameters used in various formulae. 
    double epsilon{ 0.0001 };
    double dt{ 0.001 };

	double viscousDamp{ 3.769911184308 }; // ???
	double temperature{ 300.0 };
	double kB{ 1.3806488e-8 };

    // LJ force parameters.
    double U_II{ 0.49 };
    double K_II{ 0.31 };
    double W_II{ 0.15 };
    double G_II{ 1.25 };
    double L_II{ 1.56 };
    double U_MI{ 0.78 };
    double K_MI{ 0.13 };
    double W_MI{ 0.0 };
    double G_MI{ 1.0 };
    double L_MI{ 1.56 };

};


class PlatletSystem {
public:
    MembraneNode memNode;
    Node intNode;
    SpringEdge springEdge;
    SimulationParams simulationParams;
    GeneralParams generalParams;

    std::shared_ptr<PlatletStorage> pltStorage;

public:

    PlatletSystem();

    void assignPltStorage( std::shared_ptr<PlatletStorage> );


    void initializePlatletSystem(
        thrust::host_vector<double> &host_pos_x,
        thrust::host_vector<double> &host_pos_y,
        thrust::host_vector<double> &host_pos_z,    
        thrust::host_vector<bool> &host_isFixed,
        thrust::host_vector<unsigned> &host_nodeID_L,
        thrust::host_vector<unsigned> &host_nodeID_R,
        thrust::host_vector<double> &host_len_0);


    void solvePltSystem();


    void solvePltForces();


    void setNodes(
        thrust::host_vector<double> &host_pos_x,
        thrust::host_vector<double> &host_pos_y,
        thrust::host_vector<double> &host_pos_z,    
        thrust::host_vector<bool> &host_isFixed);


    void setSpringEdge(
        thrust::host_vector<unsigned> &host_nodeID_L,
        thrust::host_vector<unsigned> &host_nodeID_R,
        thrust::host_vector<double> &host_len_0
    );


    void printPoints();


    void printConnections();
    

    void printForces();


    void printParams();

};


#endif