// ************************************
// Included for printing test messages.
#include <iostream>
#include <cstdio>
// ************************************
//#include "PlatletStorage.h"
#include "PlatletSystem.h" 
//#include "Advance_Positions.h"
//#include "Spring_Force.h"


PlatletSystem::PlatletSystem() {};

void PlatletSystem::assignPltStorage(std::shared_ptr<PlatletStorage> _pltStorage) {
	pltStorage = _pltStorage;
}


void PlatletSystem::initializePltSystem() {

    generalParams.springEdgeCount = E;
    generalParams.memNodeCount = N;

    setPltNodes();

    printPoints();

    setPltSpringEdge();



    // printConnections();

}


void PlatletSystem::solvePltSystem() {


    while (generalParams.runSim == true) {

        generalParams.iterationCounter += 1;
        generalParams.currentTime += generalParams.dt;

        solvePltForces(); // Reset Forces to zero, then solve for next time step

        Advance_Positions(node, generalParams);

        if (generalParams.iterationCounter % 10 == 0) {

            pltStorage->print_VTK_File(); 

            // Temporary just to verify the output.
            printPoints();
        }

        // Hard cap on the number of simulation steps. 
        // Currently the only way to stop the simulation.
        if (generalParams.iterationCounter >= 500) {
            generalParams.runSim = false;
        }
    }
    
}


void PlatletSystem::solvePltForces() {

    // Reset forces to zero.
    thrust::fill(node.force_x.begin(), node.force_x.end(), 0.0);    
    thrust::fill(node.force_y.begin(), node.force_y.end(), 0.0);    
    thrust::fill(node.force_z.begin(), node.force_z.end(), 0.0);


    SpringForce(node, springEdge, generalParams);

    // Used only for debugging.
    printForces();


    // LJ_Force(node);


    // Fixed nodes. Set forces to zero.
}


void PlatletSystem::setPltNodes() {

    
    // Hard coded values for now. Later initialize points randomly or in a circle.

    thrust::host_vector<double> host_pos_x(generalParams.memNodeCount);
    thrust::host_vector<double> host_pos_y(generalParams.memNodeCount);
    thrust::host_vector<double> host_pos_z(generalParams.memNodeCount);

    host_pos_x[0] = 0.0;
    host_pos_y[0] = -1.0;
    host_pos_z[0] = 0.0;

    host_pos_x[1] = 0.0;
    host_pos_y[1] = 0.0;
    host_pos_z[1] = 0.0;

    host_pos_x[2] = 1.0;
    host_pos_y[2] = 0.0;
    host_pos_z[2] = 0.0;

    host_pos_x[3] = 0.0;
    host_pos_y[3] = 1.0;
    host_pos_z[3] = 0.0;

    host_pos_x[4] = 1.0;
    host_pos_y[4] = 2.0;
    host_pos_z[4] = 0.0;


    node.pos_x.resize(generalParams.memNodeCount);
    node.pos_y.resize(generalParams.memNodeCount);
    node.pos_z.resize(generalParams.memNodeCount);

    node.velocity.resize(generalParams.memNodeCount);
    /* node.vel_x.resize(generalParams.memNodeCount);
    node.vel_y.resize(generalParams.memNodeCount);
    node.vel_z.resize(generalParams.memNodeCount); */

    node.force_x.resize(generalParams.memNodeCount);
    node.force_y.resize(generalParams.memNodeCount);
    node.force_z.resize(generalParams.memNodeCount);

    node.isFixed.resize(generalParams.memNodeCount);



    thrust::copy(host_pos_x.begin(), host_pos_x.end(), node.pos_x.begin());
    thrust::copy(host_pos_y.begin(), host_pos_y.end(), node.pos_y.begin());
    thrust::copy(host_pos_z.begin(), host_pos_z.end(), node.pos_z.begin());

    thrust::fill(node.velocity.begin(), node.velocity.end(), 0.0);

    /* thrust::fill(node.vel_x.begin(), node.vel_x.end(), 0.0);
    thrust::fill(node.vel_y.begin(), node.vel_y.end(), 0.0);
    thrust::fill(node.vel_z.begin(), node.vel_z.end(), 0.0); */

    thrust::fill(node.force_x.begin(), node.force_x.end(), 0.0);
    thrust::fill(node.force_y.begin(), node.force_y.end(), 0.0);
    thrust::fill(node.force_z.begin(), node.force_z.end(), 0.0);

    thrust::fill(node.isFixed.begin(), node.isFixed.end(), false);
}


void PlatletSystem::setPltSpringEdge() {

    /* For now we multiply by 2 since each node will be connected to exactly 2 other nodes.
    This should be later generalized to not count the springs, but to count the number of connections at each node. */
    springEdge.nodeConnections.resize(generalParams.memNodeCount * generalParams.maxConnectedSpringCount);
    springEdge.len_0.resize(generalParams.memNodeCount * generalParams.maxConnectedSpringCount);
    thrust::fill(springEdge.len_0.begin(), springEdge.len_0.end(), 3.0);


    // Set edge connections via circular connection
    for(int i = 0; i < generalParams.springEdgeCount; ++i) {
        // i-th spring connects to the i-th node.
        springEdge.nodeConnections[i*2] = (i-1 < 0) ? generalParams.memNodeCount-1 : i-1;

        // i-th spring edge connects to the (i+1)th node, except for the last node, which connects to the 0-th node.
        springEdge.nodeConnections[i*2 + 1] = (i+1 >= generalParams.springEdgeCount) ? 0 : i+1;
    }

    // TO-DO Set length_0!!!
}

void PlatletSystem::printPoints() {
    std::cout << "Testing initialization of vector position:" << std::endl;
        for(auto i = 0; i < node.pos_x.size(); ++i) {
            std::cout << "Node " << i << ": ("
                << node.pos_x[i] << ", "
                << node.pos_y[i] << ", "
                << node.pos_z[i] << ")" << std::endl;
        }
}


void PlatletSystem::printConnections() {
    std::cout << "Testing edge connections:" << std::endl;
        for(auto i = 0; i < generalParams.springEdgeCount; ++i) {
            std::cout << "Node " << i << " is connected to: "
                << springEdge.nodeConnections[i*2] << ", "
                << springEdge.nodeConnections[i*2 + 1] << std::endl;
        }

    std::cout << "Testing edge connection vector creation:" << std::endl;
    for (auto s : springEdge.nodeConnections) {
        std::cout << s << std::endl;
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