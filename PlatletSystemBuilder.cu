//********************************
#include <cuda.h>
#include <cstdlib>
#include <random>
#include <set>
#include <list>
#include <vector>
#include <memory>
#include "PlatletSystemBuilder.h"
#include "PlatletSystem.h"
# define M_PI 3.14159265358979323846  /* pi */
//********************************


PlatletSystemBuilder::PlatletSystemBuilder(double _epsilon, double _dt):
	epsilon(_epsilon), dt(_dt) {
		generalParams.epsilon = epsilon;
		generalParams.dt = dt;
	};

PlatletSystemBuilder::~PlatletSystemBuilder() {
}

unsigned PlatletSystemBuilder::addMembraneNode(glm::dvec3 pos) {

	nodePosition.push_back(pos);

	pos_x.push_back(pos.x);
	pos_y.push_back(pos.y);
	pos_z.push_back(pos.z);
	
	// Include this part with create_system_on_device().
	// node.isFixed.push_back(false);

	return pos_x.size();

}

void PlatletSystemBuilder::printNodes() {
	std::cout << "Testing initialization of vector position:\n";
	for(auto i = 0; i < pos_x.size(); ++i) {
		std::cout << "Node " << i << ": ("
			<< pos_x[i] << ", "
			<< pos_y[i] << ", "
			<< pos_z[i] << ")\n";
	}
}


unsigned PlatletSystemBuilder::addMembraneEdge(unsigned n1, unsigned n2) {

	double length = glm::length(nodePosition[n1] - nodePosition[n2]);

	len_0.push_back(length);
	nodeID_L.push_back(n1);
	nodeID_R.push_back(n2);

	return nodeID_L.size();
}


void PlatletSystemBuilder::printEdges() {
	std::cout << "Testing initialization of edge connections:\n";
	for(auto i = 0; i < nodeID_L.size(); ++i) {
		std::cout << "Edge " << i << " connecting: "
			<< nodeID_L[i] << " and "
			<< nodeID_R[i] << '\n';
	}
}

void PlatletSystemBuilder::fixNode(unsigned id) {
	isFixed[id] = true;
}


std::shared_ptr<PlatletSystem> PlatletSystemBuilder::Create_Platlet_System_On_Device() {

	// *****************************************************
	// Create and initialize (the pointer to) the final system on device.
	 
	std::shared_ptr<PlatletSystem> ptr_Platlet_System_Host = std::make_shared<PlatletSystem>();


	// *****************************************************
	// Calculations of parameter values based on vector info (size, max, etc.)
	if ( (pos_x.size() == pos_y.size()) && (pos_y.size() == pos_z.size()) ) {
		generalParams.memNodeCount = pos_x.size();
		ptr_Platlet_System_Host->generalParams.memNodeCount = generalParams.memNodeCount;
	} else {
		std::cout << "ERROR: Position vectors not all the same size.\n";
		return nullptr;
	}

	if (nodeID_L.size() == nodeID_R.size()) {
		generalParams.springEdgeCount = nodeID_L.size();
		ptr_Platlet_System_Host->generalParams.springEdgeCount = generalParams.springEdgeCount;

	} else {
		std::cout << "ERROR: Missing entry on membrane edge connection.\n";
		return nullptr;
	}

	// Temporary value for 2D.
	// Not sure what this should be in general.
	ptr_Platlet_System_Host->generalParams.maxConnectedSpringCount = 2;
	
	/* ptr_Platlet_System_Host->generalParams.epsilon = generalParams.epsilon;
	ptr_Platlet_System_Host->generalParams.dt = generalParams.dt;
	ptr_Platlet_System_Host->generalParams.viscousDamp = generalParams.viscousDamp;
	ptr_Platlet_System_Host->generalParams.temperature = generalParams.temperature;
	ptr_Platlet_System_Host->generalParams.kB = generalParams.kB;
	ptr_Platlet_System_Host->generalParams.memNodeMass = generalParams.memNodeMass; */

	ptr_Platlet_System_Host->generalParams = generalParams;

	// *****************************************************

	ptr_Platlet_System_Host->initializePlatletSystem(
		pos_x,
		pos_y,
		pos_y,
		isFixed,
		nodeID_L,
		nodeID_R,
		len_0);

	

	return ptr_Platlet_System_Host; 

	// return nullptr;
}




