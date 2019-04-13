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

    double epsilon, dt;

    PlatletSystemBuilder(double _epsilon, double _dt);
	~PlatletSystemBuilder();

    std::vector<glm::dvec3> nodePosition;
    // Host version of Node.
    thrust::host_vector<double> pos_x;
    thrust::host_vector<double> pos_y;
    thrust::host_vector<double> pos_z;
    
    thrust::host_vector<bool> isFixed;


    // Host version of SpringEdge.
    thrust::host_vector<unsigned> nodeID_L;
    thrust::host_vector<unsigned> nodeID_R;

    thrust::host_vector<double> len_0;



    GeneralParams generalParams;

    

    // Bunch of methods for adding to vectors from XML.
    unsigned addMembraneNode(glm::dvec3 pos);
    
    unsigned addMembraneEdge(unsigned n1, unsigned n2);

    void printNodes();

    void printEdges();

    // Final goal of SystemBuilder is to create a copy of the system on device.
    std::shared_ptr<PlatletSystem> Create_Platlet_System_On_Device();
};


#endif