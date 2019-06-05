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

// #include <thrust/system_error.h>
// #include <thrust/system/cuda/error.h>
// #include <sstream>


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
    // We want a picture of the initial system.
    pltStorage->print_VTK_File();

    // Make sure the bucket scheme is set so we can use it.
    setBucketScheme();
    
    // Main loop.
    // Maybe later add some way to have adaptive control
    // of the loop so that it only runs for a few time steps
    // or until it reaches equilibrium, etc.

    while (simulationParams.runSim == true) {
        simulationParams.iterationCounter += 1;
        simulationParams.currentTime += generalParams.dt;

        // Reset the bucket scheme every ten steps.
        if (simulationParams.iterationCounter % 10 == 0) {
            setBucketScheme();
        }

        // Reset Forces to zero, then solve for next time step.
        solvePltForces(); 

        Advance_Positions(node, generalParams);

        if (simulationParams.iterationCounter % simulationParams.printFileStepSize == 0) {
            pltStorage->print_VTK_File(); 
        }

        // Hard cap on the number of simulation steps. 
        // Currently the only way to stop the simulation.
        if (simulationParams.iterationCounter >= simulationParams.maxIterations) {
            simulationParams.runSim = false;
        }
    }    
}


void PlatletSystem::solvePltForces() {

    // Reset forces to zero.
    thrust::fill(node.force_x.begin(), node.force_x.end(), 0.0);    
    thrust::fill(node.force_y.begin(), node.force_y.end(), 0.0);    
    thrust::fill(node.force_z.begin(), node.force_z.end(), 0.0);

    Spring_Force(node, springEdge, generalParams);
    LJ_Force(node, bucketScheme, generalParams);

    // Used only for debugging.
    // printForces();
}


void PlatletSystem::setNodes(
    thrust::host_vector<double> &host_pos_x,
    thrust::host_vector<double> &host_pos_y,
    thrust::host_vector<double> &host_pos_z,    
    thrust::host_vector<bool> &host_isFixed) {

    // pre-allocating for speed.
    if ( node.total_count == host_pos_x.size() ) {
        node.pos_x.resize(node.total_count);
        node.pos_y.resize(node.total_count);
        node.pos_z.resize(node.total_count);

        node.velocity.resize(node.total_count);
        
        node.force_x.resize(node.total_count);
        node.force_y.resize(node.total_count);
        node.force_z.resize(node.total_count);

        node.type.resize(node.total_count);


        // Filling with values from the SystemBuilder.
        // host_vectors for PlatletSystemBuilder contain all memNodes, followed by all intNodes.
        thrust::copy(host_pos_x.begin(), host_pos_x.end(), node.pos_x.begin());
        thrust::copy(host_pos_y.begin(), host_pos_y.end(), node.pos_y.begin());
        thrust::copy(host_pos_z.begin(), host_pos_z.end(), node.pos_z.begin());

        thrust::fill(node.velocity.begin(), node.velocity.end(), 0.0);

        thrust::fill(node.force_x.begin(), node.force_x.end(), 0.0);
        thrust::fill(node.force_y.begin(), node.force_y.end(), 0.0);
        thrust::fill(node.force_z.begin(), node.force_z.end(), 0.0);

        thrust::fill(node.type.begin(), node.type.begin() + node.membrane_count, 1);
        thrust::fill(node.type.begin() + node.membrane_count, node.type.begin() + node.membrane_count + node.interior_count, 2);
    } else {
        std::cerr << "ERROR: position vector not same size as node.total_count." << std::endl;
        return; 
    }
    
    if ( node.membrane_count == host_isFixed.size() ) {
        node.isFixed.resize(node.membrane_count);
        thrust::copy(
            host_isFixed.begin(), 
            host_isFixed.begin() + node.membrane_count, 
            node.isFixed.begin());
    } else {
        std::cerr << "ERROR: isFixed vector not same size as node.membrane_count." << std::endl;
        return; 
    }
}


void PlatletSystem::setSpringEdge(
    thrust::host_vector<unsigned> &host_nodeID_L,
    thrust::host_vector<unsigned> &host_nodeID_R,
    thrust::host_vector<double> &host_len_0) {

    
    springEdge.nodeID_L.resize(springEdge.count);
    springEdge.nodeID_R.resize(springEdge.count);
    springEdge.len_0.resize(springEdge.count);

    thrust::copy(host_nodeID_L.begin(), host_nodeID_L.end(), springEdge.nodeID_L.begin());
    thrust::copy(host_nodeID_R.begin(), host_nodeID_R.end(), springEdge.nodeID_R.begin());
    thrust::copy(host_len_0.begin(), host_len_0.end(), springEdge.len_0.begin());
    
    
    node.connectedSpringID.resize(node.membrane_count * node.maxConnectedSpringCount);
    node.connectedSpringCount.resize(node.membrane_count);

    thrust::fill(node.connectedSpringCount.begin(), node.connectedSpringCount.end(), 0);
    thrust::fill(node.connectedSpringID.begin(), node.connectedSpringID.end(), 0);


    // std::cout << "Building connectedSpringID vector.\n";
    // Merge nodeID_(R/L) into connectedSpringID using maxConnectedSpringCount.
    // Build connectedSpringCount as we go.
    for (auto s = 0; s < springEdge.count; ++s) {
        for (auto isLeft = 0; isLeft < 2; ++isLeft) {
            unsigned n;
            if (isLeft == 0) {
                n = springEdge.nodeID_R[s];
            } else if (isLeft == 1) {
                n = springEdge.nodeID_L[s];
            }
            unsigned index = n * node.maxConnectedSpringCount + node.connectedSpringCount[n];
            node.connectedSpringID[index] = s;
            ++node.connectedSpringCount[n];
        }
    }
}


void PlatletSystem::printPoints() {
    std::cout << "Testing initialization of vector position:\n";
    for(auto i = 0; i < node.pos_x.size(); ++i) {
        std::cout << "Node " << i << ": ("
            << node.pos_x[i] << ", "
            << node.pos_y[i] << ", "
            << node.pos_z[i] << ")\n";
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
    for(auto i = 0; i <  node.connectedSpringID.size(); ++i) {
        std::cout << node.connectedSpringID[i] << '\n';
    } */

    std::cout << "Testing nodeDegree vector:\n";
    for(auto i = node.connectedSpringCount.begin(); i != node.connectedSpringCount.end(); ++i) {
        std::cout << *i << '\n';
    }
}


void PlatletSystem::printForces() {
    std::cout << "Testing force calculation:" << std::endl;
        for(auto i = 0; i < node.force_x.size(); ++i) {
            std::cout << "Force on node " << i << ": ("
                << node.force_x[i] << ", "
                << node.force_y[i] << ", "
                << node.force_z[i] << ")" << std::endl;
        }
}


void PlatletSystem::printParams() {
    std::cout << "Testing parameter initialization: \n";

    // std::cout << "epsilon: " << generalParams.epsilon << '\n';
    // std::cout << "dt: " << generalParams.dt << '\n';
    // std::cout << "viscousDamp: " << generalParams.viscousDamp << '\n';
    // std::cout << "temperature: " << generalParams.temperature << '\n';
    // std::cout << "kB: " << generalParams.kB << '\n';
    // std::cout << "memNodeMass: " << memNode.mass << '\n';
    // std::cout << "memSpringStiffness: " << springEdge.stiffness << '\n';
    // std::cout << "Morse U: " << generalParams.U_II << '\n';
    // std::cout << "Morse P: " << generalParams.P_II << '\n';
    // std::cout << "Morse R_eq: " << generalParams.R_eq_II << '\n';
    std::cout << "memNodeCount: " << node.membrane_count << '\n';
    std::cout << "intNodeCount: " << node.interior_count << '\n';

}


void PlatletSystem::setBucketScheme() {
    
    initialize_bucket_dimensions(
        node,
        domainParams);
    
    set_bucket_grids(
        node,
        domainParams,
        bucketScheme);

    assign_nodes_to_buckets(
        node,
        domainParams,
        bucketScheme);
    
    extend_to_bucket_neighbors(
        node,
        domainParams,
        bucketScheme);
}


void PlatletSystem::initialize_bucket_vectors() {

    bucketScheme.bucket_ID.resize(node.total_count);
    bucketScheme.globalNode_ID.resize(node.total_count);
    bucketScheme.bucket_ID_expanded.resize( node.total_count * 27 );
    bucketScheme.globalNode_ID_expanded.resize( node.total_count * 27 );

}