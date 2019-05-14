// ************************************
// Included for printing test messages.
#include <iostream>
#include <cstdio>
// ************************************
#include "PlatletStorage.h"
#include "PlatletSystem.h" 
#include "Advance_Positions.h"
#include "Spring_Force.h"


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
    

    setMembraneNodes(
        host_pos_x,
        host_pos_y,
        host_pos_z,    
        host_isFixed);

    // printPoints();

    setSpringEdge(
        host_nodeID_L,
        host_nodeID_R,
        host_len_0);



    // printConnections();

    printParams();

}


void PlatletSystem::solvePltSystem() {


    while (simulationParams.runSim == true) {

        simulationParams.iterationCounter += 1;
        simulationParams.currentTime += generalParams.dt;

        // Reset Forces to zero, then solve for next time step.
        solvePltForces(); 

        Advance_Positions(node, generalParams);

        if (simulationParams.iterationCounter % 50 == 0) {

            pltStorage->print_VTK_File(); 

            // Temporary just to verify the output.
            // printPoints();
        }

        // Hard cap on the number of simulation steps. 
        // Currently the only way to stop the simulation.
        if (simulationParams.iterationCounter >= 5000) {
            simulationParams.runSim = false;
        }

       // simulationParams.runSim = false;
    }
    
}


void PlatletSystem::solvePltForces() {

    // Reset forces to zero.
    thrust::fill(memNode.force_x.begin(), memNode.force_x.end(), 0.0);    
    thrust::fill(memNode.force_y.begin(), memNode.force_y.end(), 0.0);    
    thrust::fill(memNode.force_z.begin(), memNode.force_z.end(), 0.0);


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
   
    memNode.pos_x.resize(generalParams.memNodeCount);
    memNode.pos_y.resize(generalParams.memNodeCount);
    memNode.pos_z.resize(generalParams.memNodeCount);

    memNode.velocity.resize(generalParams.memNodeCount);
    
    memNode.force_x.resize(generalParams.memNodeCount);
    memNode.force_y.resize(generalParams.memNodeCount);
    memNode.force_z.resize(generalParams.memNodeCount);

    memNode.isFixed.resize(generalParams.memNodeCount);


    thrust::copy(host_pos_x.begin(), host_pos_x.end(), memNode.pos_x.begin());
    thrust::copy(host_pos_y.begin(), host_pos_y.end(), memNode.pos_y.begin());
    thrust::copy(host_pos_z.begin(), host_pos_z.end(), memNode.pos_z.begin());

    thrust::fill(memNode.velocity.begin(), memNode.velocity.end(), 0.0);

    thrust::fill(memNode.force_x.begin(), memNode.force_x.end(), 0.0);
    thrust::fill(memNode.force_y.begin(), memNode.force_y.end(), 0.0);
    thrust::fill(memNode.force_z.begin(), memNode.force_z.end(), 0.0);

    thrust::copy(host_isFixed.begin(), host_isFixed.end(), memNode.isFixed.begin());
}


void PlatletSystem::setSpringEdge(
    thrust::host_vector<unsigned> &host_nodeID_L,
    thrust::host_vector<unsigned> &host_nodeID_R,
    thrust::host_vector<double> &host_len_0) {

    springEdge.nodeID_L.resize(generalParams.springEdgeCount);
    springEdge.nodeID_R.resize(generalParams.springEdgeCount);
    springEdge.len_0.resize(generalParams.springEdgeCount);

    memNode.springConnections.resize(generalParams.memNodeCount * generalParams.maxConnectedSpringCount);
    memNode.numConnectedSprings.resize(generalParams.memNodeCount);

    thrust::fill(memNode.numConnectedSprings.begin(), memNode.numConnectedSprings.end(), 0);
    thrust::fill(memNode.springConnections.begin(), memNode.springConnections.end(), 47);

    thrust::copy(host_nodeID_L.begin(), host_nodeID_L.end(), springEdge.nodeID_L.begin());
    thrust::copy(host_nodeID_R.begin(), host_nodeID_R.end(), springEdge.nodeID_R.begin());
    thrust::copy(host_len_0.begin(), host_len_0.end(), springEdge.len_0.begin());

    // Merge nodeID_(R/L) into springConnections using maxConnectedSpringCount.
    // Build numConnectedSprings as we go.
    for (auto s = 0; s < generalParams.springEdgeCount; ++s) {
        unsigned n = springEdge.nodeID_L[s];
        unsigned index = n * generalParams.maxConnectedSpringCount + memNode.numConnectedSprings[n];
        memNode.springConnections[index] = s;
        ++memNode.numConnectedSprings[n];

        n = springEdge.nodeID_R[s];
        index = n * generalParams.maxConnectedSpringCount + memNode.numConnectedSprings[n];
        memNode.springConnections[index] = s;
        ++memNode.numConnectedSprings[n];
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
    std::cout << "Testing edge connections in PlatletSystem: \n";
    for(auto i = 0; i < generalParams.springEdgeCount; ++i) {
        std::cout << "Spring " << i << " is connected to: "
            << springEdge.nodeID_L[i] << ", "
            << springEdge.nodeID_R[i] 
            << " with equilibrium length " << springEdge.len_0[i] << '\n';
    }

    std::cout << "Testing springConnections vector:\n";
    for(auto i = 0; i <  memNode.springConnections.size(); ++i) {
        std::cout << memNode.springConnections[i] << '\n';
    }

    std::cout << "Testing nodeDegree vector:\n";
    for(auto i = memNode.numConnectedSprings.begin(); i != memNode.numConnectedSprings.end(); ++i) {
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
    std::cout << "memNodeMass: " << generalParams.memNodeMass << '\n';
    std::cout << "memSpringStiffness: " << generalParams.memSpringStiffness << '\n';
}