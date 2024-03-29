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

    thrust::device_vector<double> energy;

    // 1: membrane node
    // 2: interior node
    thrust::device_vector<unsigned> type;

    double membrane_mass{ 1.0 };
    double interior_mass{ 1.0 };
    unsigned membrane_count{ 0 };
    unsigned interior_count{ 0 };
    unsigned total_count{ 0 };

    // *******************************************
    // Membrane node info
    
    // size() = node.membrane_count
    thrust::device_vector<bool> isFixed;

    // Vector size is M * N, 
    // where    M = maxConnectedSpringCount
    // and      N = memNodeCount.
    // Each entry corresponds to the ID of 
    // a spring that node is connected to.
    thrust::device_vector<unsigned> connectedSpringID;

    // size() = node.membrane_count
    thrust::device_vector<unsigned> connectedSpringCount;

    // Used for indexing purposes.
    unsigned maxConnectedSpringCount{ 20 };
    // *******************************************
};

struct SimulationParams {

    // For tracking the simulation while it is running. 
    bool runSim{ true };
    double currentTime{ 0.0 };
    unsigned iterationCounter{ 0 };
    unsigned maxIterations{ 40000 };
    unsigned printFileStepSize{ 500 };

};


struct GeneralParams {

    // Parameters used in various formulae. 
    double epsilon{ 0.0001 };
    double dt{ 0.0005 };

	double viscousDamp{ 3.769911184308 }; // ???
	double temperature{ 300.0 };
	double kB{ 1.3806488e-8 };

    double totalEnergy{ 0.0 };

    // LJ force parameters.
    /* double U_II{ 0.049 };
    double K_II{ 0.31 };
    double W_II{ 0.015 };
    double G_II{ 1.25 };
    double L_II{ 0.36 };
    double U_MI{ 0.078 };
    double K_MI{ 0.13 };
    double W_MI{ 0.0 };
    double G_MI{ 1.0 };
    double L_MI{ 0.36 }; */

    double U_II{ 1.0 };
    double P_II{ 2.0 };
    // 3d
    // double R_eq_II{ 0.271441761659491 };
    // 2d
    double R_eq_II{ 0.12 };



    double U_MI{ 1.0 };
    double P_MI{ 2.0 };
    // 3d
    // double R_eq_MI{ 0.271441761659491 };
    // 2d
    double R_eq_MI{ 0.12 };

};


struct DomainParams {

    // For now, these are scalars, but in future these may be vectors to have one for each platelet.
    double min_x;
    double min_y;
    double min_z;
    double max_x;
    double max_y;
    double max_z;

    double gridSpacing{ 0.200 };

    unsigned bucketCount_x;
    unsigned bucketCount_y;
    unsigned bucketCount_z;
    unsigned bucketCount_total{ 0 };
    
};


struct BucketScheme {

    thrust::device_vector<unsigned> keyBegin;
    thrust::device_vector<unsigned> keyEnd;

    thrust::device_vector<unsigned> bucket_ID;
    thrust::device_vector<unsigned> globalNode_ID;

    thrust::device_vector<unsigned> bucket_ID_expanded;
    thrust::device_vector<unsigned> globalNode_ID_expanded;

    unsigned endIndexBucketKeys;


};


class PlatletSystem {
public:
    Node node;
    SpringEdge springEdge;
    SimulationParams simulationParams;
    GeneralParams generalParams;
    DomainParams domainParams;
    BucketScheme bucketScheme;
    

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


    void setBucketScheme();


    void initialize_bucket_vectors();

};


#endif