// ************************************
// Included for printing test messages.
#include <iostream>
#include <cstdio>
// ************************************
#include "PlatletSystem.h" 

#include "Advance_Positions.h"
#include "Spring_Force.h"


PlatletSystem::PlatletSystem() {};


void PlatletSystem::initializePltSystem(unsigned N) {

    setPltNodes(N);

    printPoints();

    setPltSpringEdge(N);

    printConnections();

}


void PlatletSystem::solvePltSystem() {


    // Advance_Positions();

    solvePltForces(); // Reset Forces to zero, then solve for next time step

    // Output stuff to file.

    // 

}


void PlatletSystem::solvePltForces() {

    Spring_Force(node, springEdge);

    printForces();

    //LJ_Force(node);


}


void PlatletSystem::setPltNodes(unsigned N) {

    
    // Hard coded values for now. Later initialize points randomly or in a circle.

    thrust::host_vector<double> host_pos_x(3);
    thrust::host_vector<double> host_pos_y(3);
    thrust::host_vector<double> host_pos_z(3);

    host_pos_x[0] = -1.0f;
    host_pos_y[0] = 0.0f;
    host_pos_z[0] = 0.0f;

    /* node.vel_x[0] = 0.0;
    node.vel_y[0] = 0.0;
    node.vel_z[0] = 0.0; */

    host_pos_x[1] = 1.0f;
    host_pos_y[1] = 0.0f;
    host_pos_z[1] = 0.0f;

   /*  node.vel_x[1] = 0.0;
    node.vel_y[1] = 0.0;
    node.vel_z[1] = 0.0; */

    host_pos_x[2] = 0.0f;
    host_pos_y[2] = 1.0f;
    host_pos_z[2] = 0.0f;

    /* node.vel_x[2] = 0.0;
    node.vel_y[2] = 0.0;
    node.vel_z[2] = 0.0; */




    // TO-DO: change N to generalParams.maxPltNodeCount

    node.pos_x.resize(N);
    node.pos_y.resize(N);
    node.pos_z.resize(N);

    node.vel_x.resize(N);
    node.vel_y.resize(N);
    node.vel_z.resize(N);

    node.force_x.resize(N);
    node.force_y.resize(N);
    node.force_z.resize(N);

    thrust::fill(node.vel_x.begin(), node.vel_x.end(), 0.0);
    thrust::fill(node.vel_y.begin(), node.vel_y.end(), 0.0);
    thrust::fill(node.vel_z.begin(), node.vel_z.end(), 0.0);

    thrust::fill(node.force_x.begin(), node.force_x.end(), 0.0);
    thrust::fill(node.force_y.begin(), node.force_y.end(), 0.0);
    thrust::fill(node.force_z.begin(), node.force_z.end(), 0.0);

    thrust::copy(host_pos_x.begin(), host_pos_x.end(), node.pos_x.begin());
    thrust::copy(host_pos_y.begin(), host_pos_y.end(), node.pos_y.begin());
    thrust::copy(host_pos_z.begin(), host_pos_z.end(), node.pos_z.begin());    
    //Fill device vectors with test values.
    

    /* pos_x[3] = 0.0f;
    pos_y[3] = 3.0f;
    pos_z[3] = 0.0f;

    vel_x[3] = 0.0;
    vel_y[3] = 0.0;
    vel_z[3] = 0.0; */
}


void PlatletSystem::setPltSpringEdge(unsigned N) {
    // TO-DO: change N to generalParams.maxPltSpringEdgeCount

    springEdge.node_L.resize(N);
    springEdge.node_R.resize(N);


    // Set edge values (connections, length) via circular connection
    for(int i = 0; i < N; ++i) {
        springEdge.node_L[i] = i;
        springEdge.node_R[i] = (i+1 >= springEdge.node_L.size()) ? 0 : i+1;
    }
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
        for(auto i = 0; i < springEdge.node_L.size(); ++i) {
            std::cout << "Edge " << i << ": "
                << springEdge.node_L[i] << ", "
                << springEdge.node_R[i] << std::endl;
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