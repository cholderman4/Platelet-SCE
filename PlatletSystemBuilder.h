#ifndef PLATLET_SYSTEM_BUILDER_H_
#define PLATLET_SYSTEM_BUILDER_H_

#include "PlatletSystem.h"

//********************************
// Not sure if all these are needed.
#include <thrust/host_vector.h>
#include "glm/glm.hpp"
#include <list>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
//********************************

// Temporary fix until I can figure out GLM.
/* struct vec3 {
	double x, y, z;
}; */

class PlatletSystemBuilder {
public:

    PlatletSystemBuilder(double _epsilon, double _dt);
	~PlatletSystemBuilder();

    double epsilon, dt;


    // Host version of Node.
    thrust::host_vector<double> pos_x;
    thrust::host_vector<double> pos_y;
    thrust::host_vector<double> pos_z;
    
    thrust::host_vector<double> velocity;

    thrust::host_vector<double> force_x;
    thrust::host_vector<double> force_y;
    thrust::host_vector<double> force_z;

    thrust::host_vector<bool> isFixed;


    // Host version of SpringEdge.
    thrust::host_vector<unsigned> nodeConnections;

    thrust::host_vector<double> len_0;



    GeneralParams generalParams;

    

    // Bunch of methods for adding to vectors from XML.
    unsigned addMembraneNode(glm::dvec3 pos);
    
    unsigned addMembraneEdge();

    void printNodes();

    // Final goal of SystemBuilder is to create a copy of the system on device.
    std::shared_ptr<PlatletSystem> Create_Platlet_System_On_Device();
};


#endif