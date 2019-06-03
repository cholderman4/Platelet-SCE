// ************************************
// Included for printing test messages.
#include <iostream>
#include <cstdio>
// ************************************
#include "PlatletStorage.h"
#include "PlatletSystem.h" 
#include "Advance_Positions.h"
#include "Spring_Force.h"
#include "LJ_Force.h"
#include "Bucket_Sort.h"


PlatletSystem::PlatletSystem() {};

void PlatletSystem::assignPltStorage(std::shared_ptr<PlatletStorage> _pltStorage) {
	pltStorage = _pltStorage;
}


void PlatletSystem::initializePlatletSystem(
    thrust::host_vector<double> &host_pos_x,
    thrust::host_vector<double> &host_pos_y,
    thrust::host_vector<double> &host_pos_z,    
    thrust::host_vector<bool> &host_isFixed,
    thrust::host_vector<unsigned> &host_nodeID_L,
    thrust::host_vector<unsigned> &host_nodeID_R,
    thrust::host_vector<double> &host_len_0) {
    
// std::cerr << "Initializing...\n";

// std::cerr << "Initializing nodes\n";
    setNodes(
        host_pos_x,
        host_pos_y,
        host_pos_z,    
        host_isFixed);


    initialize_bucket_vectors();


    // printPoints();

// std::cerr << "Initializing edges\n";
    setSpringEdge(
        host_nodeID_L,
        host_nodeID_R,
        host_len_0);



    // printConnections();

// std::cerr << "Printing parameters.\n";
    printParams();




}


void PlatletSystem::solvePltSystem() {
// std::cerr << "solvePltSystem\n";
    // We want a picture of the initial system.
    pltStorage->print_VTK_File();

// std::cerr << "setBucketScheme\n";
    // Make sure the bucket schem is set so we can use it.
    setBucketScheme();
    
    // Main loop.
    // Maybe later add some way to have adaptive control
    // of the loop so that it only runs for a few time steps
    // or until it reaches equilibrium, etc.

// std::cerr << "starting main loop\n";
    while (simulationParams.runSim == true) {
        simulationParams.iterationCounter += 1;
        simulationParams.currentTime += generalParams.dt;
// std::cerr << "in main loop: " << simulationParams.iterationCounter << " iterations\n";

// std::cerr << "checking bucketScheme reset\n";
        // Reset the bucket scheme every ten steps.
        if (simulationParams.iterationCounter % 10 == 0) {
// std::cerr << "resetting bucketScheme after 10 steps\n";
            setBucketScheme();
        }

// std::cerr << "solvePltForces()\n";
        // Reset Forces to zero, then solve for next time step.
        solvePltForces(); 

std::cerr << "Advance_Positions(memNode)\n";
        Advance_Positions(memNode, generalParams);
std::cerr << "Advance_Positions(intNode)\n";
        Advance_Positions(intNode, generalParams);

// std::cerr << "checking: print_VTK_File\n";
        if (simulationParams.iterationCounter % simulationParams.printFileStepSize == 0) {
            pltStorage->print_VTK_File(); 
        }

// std::cerr << "Checking: simulation end\n";
        // Hard cap on the number of simulation steps. 
        // Currently the only way to stop the simulation.
        if (simulationParams.iterationCounter >= simulationParams.maxIterations) {
            simulationParams.runSim = false;
        }
    }    
}


void PlatletSystem::solvePltForces() {

    // Reset forces to zero.
    thrust::fill(memNode.force_x.begin(), memNode.force_x.end(), 0.0);    
    thrust::fill(memNode.force_y.begin(), memNode.force_y.end(), 0.0);    
    thrust::fill(memNode.force_z.begin(), memNode.force_z.end(), 0.0);

    thrust::fill(intNode.force_x.begin(), intNode.force_x.end(), 0.0);    
    thrust::fill(intNode.force_y.begin(), intNode.force_y.end(), 0.0);    
    thrust::fill(intNode.force_z.begin(), intNode.force_z.end(), 0.0);


    Spring_Force(memNode, springEdge, generalParams);

    // Used only for debugging.
    // printForces();


    LJ_Force(memNode, intNode, bucketScheme, generalParams);


    // Fixed nodes. Set forces to zero.
}


void PlatletSystem::setNodes(
    thrust::host_vector<double> &host_pos_x,
    thrust::host_vector<double> &host_pos_y,
    thrust::host_vector<double> &host_pos_z,    
    thrust::host_vector<bool> &host_isFixed) {

    // pre-allocating for speed.
    memNode.pos_x.resize(memNode.count);
    memNode.pos_y.resize(memNode.count);
    memNode.pos_z.resize(memNode.count);

    memNode.velocity.resize(memNode.count);
    
    memNode.force_x.resize(memNode.count);
    memNode.force_y.resize(memNode.count);
    memNode.force_z.resize(memNode.count);

    memNode.isFixed.resize(memNode.count);

    // Filling with values from the SystemBuilder.
    // host_vectors for PlatletSystemBuilder contain all memNodes, followed by all intNodes.
    // First, we fill only the membrane nodes.
    thrust::copy(host_pos_x.begin(), host_pos_x.begin() + memNode.count, memNode.pos_x.begin());
    thrust::copy(host_pos_y.begin(), host_pos_y.begin() + memNode.count, memNode.pos_y.begin());
    thrust::copy(host_pos_z.begin(), host_pos_z.begin() + memNode.count, memNode.pos_z.begin());

    thrust::fill(memNode.velocity.begin(), memNode.velocity.begin() + memNode.count, 0.0);

    thrust::fill(memNode.force_x.begin(), memNode.force_x.begin() + memNode.count, 0.0);
    thrust::fill(memNode.force_y.begin(), memNode.force_y.begin() + memNode.count, 0.0);
    thrust::fill(memNode.force_z.begin(), memNode.force_z.begin() + memNode.count, 0.0);

    thrust::copy(host_isFixed.begin(), host_isFixed.begin() + memNode.count, memNode.isFixed.begin());

    // Now we repeat the process for the internal nodes.
    intNode.pos_x.resize(intNode.count);
    intNode.pos_y.resize(intNode.count);
    intNode.pos_z.resize(intNode.count);

    intNode.velocity.resize(intNode.count);
    
    intNode.force_x.resize(intNode.count);
    intNode.force_y.resize(intNode.count);
    intNode.force_z.resize(intNode.count);

    thrust::copy(host_pos_x.begin() + memNode.count, host_pos_x.end(), intNode.pos_x.begin());
    thrust::copy(host_pos_y.begin() + memNode.count, host_pos_y.end(), intNode.pos_y.begin());
    thrust::copy(host_pos_z.begin() + memNode.count, host_pos_z.end(), intNode.pos_z.begin());

    thrust::fill(intNode.velocity.begin(), intNode.velocity.end(), 0.0);

    thrust::fill(intNode.force_x.begin(), intNode.force_x.end(), 0.0);
    thrust::fill(intNode.force_y.begin(), intNode.force_y.end(), 0.0);
    thrust::fill(intNode.force_z.begin(), intNode.force_z.end(), 0.0);
}


void PlatletSystem::setSpringEdge(
    thrust::host_vector<unsigned> &host_nodeID_L,
    thrust::host_vector<unsigned> &host_nodeID_R,
    thrust::host_vector<double> &host_len_0) {


        
    std::cout << "Resizing springEdge vectors.\n";

    springEdge.nodeID_L.resize(springEdge.count);
    springEdge.nodeID_R.resize(springEdge.count);
    springEdge.len_0.resize(springEdge.count);

    memNode.connectedSpringID.resize(memNode.count * memNode.maxConnectedSpringCount);
    memNode.connectedSpringCount.resize(memNode.count);

    std::cout << "Filling vectors with garbage values.\n";

    thrust::fill(memNode.connectedSpringCount.begin(), memNode.connectedSpringCount.end(), 0);
    thrust::fill(memNode.connectedSpringID.begin(), memNode.connectedSpringID.end(), 47);

    // std::cout << "Copying from host to device.\n";

    // std::cout << "nodeID_L\n";
    // std::cout << "springEdge.count: " << springEdge.count 
        // << "\t host_nodeID.size: " << host_nodeID_L.size() << '\n';
    thrust::copy(host_nodeID_L.begin(), host_nodeID_L.end(), springEdge.nodeID_L.begin());
     
    // std::cout << "nodeID_R\n";
    thrust::copy(host_nodeID_R.begin(), host_nodeID_R.end(), springEdge.nodeID_R.begin());
    
    // std::cout << "len_0\n";
    thrust::copy(host_len_0.begin(), host_len_0.end(), springEdge.len_0.begin());

    // std::cout << "Building connectedSpringID vector.\n";
    // Merge nodeID_(R/L) into connectedSpringID using maxConnectedSpringCount.
    // Build connectedSpringCount as we go.
    for (auto s = 0; s < springEdge.count; ++s) {
        unsigned n = springEdge.nodeID_L[s];
        unsigned index = n * memNode.maxConnectedSpringCount + memNode.connectedSpringCount[n];
        memNode.connectedSpringID[index] = s;
        ++memNode.connectedSpringCount[n];

        n = springEdge.nodeID_R[s];
        index = n * memNode.maxConnectedSpringCount + memNode.connectedSpringCount[n];
        memNode.connectedSpringID[index] = s;
        ++memNode.connectedSpringCount[n];
    }

}


void PlatletSystem::printPoints() {
    std::cout << "Testing initialization of vector position:\n";
    for(auto i = 0; i < memNode.pos_x.size(); ++i) {
        std::cout << "Node " << i << ": ("
            << memNode.pos_x[i] << ", "
            << memNode.pos_y[i] << ", "
            << memNode.pos_z[i] << ")\n";
    }
}


void PlatletSystem::printConnections() {
    /* std::cout << "Testing edge connections in PlatletSystem: \n";
    for(auto i = 0; i < springEdge.count; ++i) {
        std::cout << "Spring " << i << " is connected to: "
            << springEdge.nodeID_L[i] << ", "
            << springEdge.nodeID_R[i] 
            << " with equilibrium length " << springEdge.len_0[i] << '\n';
    }

    std::cout << "Testing connectedSpringID vector:\n";
    for(auto i = 0; i <  memNode.connectedSpringID.size(); ++i) {
        std::cout << memNode.connectedSpringID[i] << '\n';
    } */

    std::cout << "Testing nodeDegree vector:\n";
    for(auto i = memNode.connectedSpringCount.begin(); i != memNode.connectedSpringCount.end(); ++i) {
        std::cout << *i << '\n';
    }
}


void PlatletSystem::printForces() {
    std::cout << "Testing force calculation:" << std::endl;
        for(auto i = 0; i < memNode.force_x.size(); ++i) {
            std::cout << "Force on node " << i << ": ("
                << memNode.force_x[i] << ", "
                << memNode.force_y[i] << ", "
                << memNode.force_z[i] << ")" << std::endl;
        }
}


void PlatletSystem::printParams() {
    std::cout << "Testing parameter initialization: \n";

    std::cout << "epsilon: " << generalParams.epsilon << '\n';
    std::cout << "dt: " << generalParams.dt << '\n';
    std::cout << "viscousDamp: " << generalParams.viscousDamp << '\n';
    std::cout << "temperature: " << generalParams.temperature << '\n';
    std::cout << "kB: " << generalParams.kB << '\n';
    std::cout << "memNodeMass: " << memNode.mass << '\n';
    std::cout << "memSpringStiffness: " << springEdge.stiffness << '\n';
    std::cout << "Morse U: " << generalParams.U_II << '\n';
    std::cout << "Morse P: " << generalParams.P_II << '\n';
    std::cout << "Morse R_eq: " << generalParams.R_eq_II << '\n';
}


void PlatletSystem::setBucketScheme() {
    
    std::cout << "initialize_bucket_dimensions\n";
    initialize_bucket_dimensions(
        memNode,
        intNode,
        domainParams);
    
    std::cout << "set_bucket_grids\n";
    set_bucket_grids(
        memNode,
        intNode,
        domainParams,
        bucketScheme);

    std::cout << "assign_nodes_to_buckets\n";
    assign_nodes_to_buckets(
        memNode,
        intNode,
        domainParams,
        bucketScheme);
    
    std::cout << "extend_to_bucket_neighbors\n";    
    extend_to_bucket_neighbors(
        memNode,
        intNode,
        domainParams,
        bucketScheme);
}


void PlatletSystem::initialize_bucket_vectors() {

    bucketScheme.bucket_ID.resize(memNode.count + intNode.count);
    bucketScheme.globalNode_ID.resize(memNode.count + intNode.count);
    bucketScheme.bucket_ID_expanded.resize( (memNode.count + intNode.count) * 27 );
    bucketScheme.globalNode_ID_expanded.resize( (memNode.count + intNode.count) * 27 );

}