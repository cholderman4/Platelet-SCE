// ************************************
// Included for printing test messages.
#include <iostream>
#include <cstdio>
// ************************************
//#include "PlatletStorage.h"
#include "PlatletSystem.h" 
#include "Advance_Positions.h"
#include "Spring_Force.h"


PlatletSystem::PlatletSystem() {};

/* void PlatletSystem::assignPltStorage(std::shared_ptr<PlatletStorage> _pltStorage) {
	pltStorage = _pltStorage;
} */


void PlatletSystem::initializePlatletSystem(
    thrust::host_vector<double> &host_pos_x,
    thrust::host_vector<double> &host_pos_y,
    thrust::host_vector<double> &host_pos_z,    
    thrust::host_vector<bool> &host_isFixed,
    thrust::host_vector<unsigned> &host_nodeID_L,
    thrust::host_vector<unsigned> &host_nodeID_R,
    thrust::host_vector<double> &host_len_0) {
    

    setMembraneNodes(
        host_pos_x,
        host_pos_y,
        host_pos_z,    
        host_isFixed);

    printPoints();

    setSpringEdge(
        host_nodeID_L,
        host_nodeID_R,
        host_len_0);



    printConnections();

}


void PlatletSystem::solvePltSystem() {


    while (simulationParams.runSim == true) {

        simulationParams.iterationCounter += 1;
        simulationParams.currentTime += generalParams.dt;

        solvePltForces(); // Reset Forces to zero, then solve for next time step

        Advance_Positions(node, generalParams);

        if (simulationParams.iterationCounter % 10 == 0) {

            // pltStorage->print_VTK_File(); 

            // Temporary just to verify the output.
            printPoints();
        }

        // Hard cap on the number of simulation steps. 
        // Currently the only way to stop the simulation.
        if (simulationParams.iterationCounter >= 100) {
            simulationParams.runSim = false;
        }

       // simulationParams.runSim = false;
    }
    
}


void PlatletSystem::solvePltForces() {

    // Reset forces to zero.
    thrust::fill(node.force_x.begin(), node.force_x.end(), 0.0);    
    thrust::fill(node.force_y.begin(), node.force_y.end(), 0.0);    
    thrust::fill(node.force_z.begin(), node.force_z.end(), 0.0);


    Spring_Force(node, springEdge, generalParams);

    // Used only for debugging.
    // printForces();


    // LJ_Force(node);


    // Fixed nodes. Set forces to zero.
}


void PlatletSystem::setMembraneNodes(
    thrust::host_vector<double> &host_pos_x,
    thrust::host_vector<double> &host_pos_y,
    thrust::host_vector<double> &host_pos_z,    
    thrust::host_vector<bool> &host_isFixed) {
   
    node.pos_x.resize(generalParams.memNodeCount);
    node.pos_y.resize(generalParams.memNodeCount);
    node.pos_z.resize(generalParams.memNodeCount);

    node.velocity.resize(generalParams.memNodeCount);
    
    node.force_x.resize(generalParams.memNodeCount);
    node.force_y.resize(generalParams.memNodeCount);
    node.force_z.resize(generalParams.memNodeCount);

    node.isFixed.resize(generalParams.memNodeCount);


    thrust::copy(host_pos_x.begin(), host_pos_x.end(), node.pos_x.begin());
    thrust::copy(host_pos_y.begin(), host_pos_y.end(), node.pos_y.begin());
    thrust::copy(host_pos_z.begin(), host_pos_z.end(), node.pos_z.begin());

    thrust::fill(node.velocity.begin(), node.velocity.end(), 0.0);

    thrust::fill(node.force_x.begin(), node.force_x.end(), 0.0);
    thrust::fill(node.force_y.begin(), node.force_y.end(), 0.0);
    thrust::fill(node.force_z.begin(), node.force_z.end(), 0.0);

    thrust::copy(host_isFixed.begin(), host_isFixed.end(), node.isFixed.begin());
}


void PlatletSystem::setSpringEdge(
    thrust::host_vector<unsigned> &host_nodeID_L,
    thrust::host_vector<unsigned> &host_nodeID_R,
    thrust::host_vector<double> &host_len_0) {

    springEdge.nodeID_L.resize(generalParams.springEdgeCount);
    springEdge.nodeID_R.resize(generalParams.springEdgeCount);
    springEdge.len_0.resize(generalParams.springEdgeCount);

    springEdge.nodeConnections.resize(generalParams.memNodeCount * generalParams.maxConnectedSpringCount);
    springEdge.nodeDegree.resize(generalParams.memNodeCount);

    thrust::fill(springEdge.nodeDegree.begin(), springEdge.nodeDegree.end(), 0);
    thrust::fill(springEdge.nodeConnections.begin(), springEdge.nodeConnections.end(), 47);

    thrust::copy(host_nodeID_L.begin(), host_nodeID_L.end(), springEdge.nodeID_L.begin());
    thrust::copy(host_nodeID_R.begin(), host_nodeID_R.end(), springEdge.nodeID_R.begin());
    thrust::copy(host_len_0.begin(), host_len_0.end(), springEdge.len_0.begin());

    // Merge nodeID_(R/L) into nodeConnections with maxConnectedSpringCount.
    // Build nodeDegree as we go.
    for (auto s = 0; s < generalParams.springEdgeCount; ++s) {
        unsigned node = springEdge.nodeID_L[s];
        unsigned index = node * generalParams.maxConnectedSpringCount + springEdge.nodeDegree[node];
        springEdge.nodeConnections[index] = s;
        ++springEdge.nodeDegree[node];

        node = springEdge.nodeID_R[s];
        index = node * generalParams.maxConnectedSpringCount + springEdge.nodeDegree[node];
        springEdge.nodeConnections[index] = s;
        ++springEdge.nodeDegree[node];
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
    std::cout << "Testing edge connections in PlatletSystem: \n";
    for(auto i = 0; i < generalParams.springEdgeCount; ++i) {
        std::cout << "Node " << i << " is connected to: "
            << springEdge.nodeID_L[i] << ", "
            << springEdge.nodeID_R[i] << '\n';
    }

    std::cout << "Testing nodeConnections vector:\n";
    for(auto i = 0; i <  springEdge.nodeConnections.size(); ++i) {
        std::cout << springEdge.nodeConnections[i] << '\n';
    }

    std::cout << "Testing nodeDegree vector:\n";
    for(auto i = springEdge.nodeDegree.begin(); i != springEdge.nodeDegree.end(); ++i) {
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