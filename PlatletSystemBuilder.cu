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
	epsilon(_epsilon), dt(_dt) {}

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
	nodeConnections.push_back(n1);
	nodeConnections.push_back(n2);

	return nodeConnections.size();
}


void PlatletSystemBuilder::printEdges() {
	std::cout << "Testing initialization of edge connections:\n";
	for(auto i = 0; i < nodeConnections.size()/2; ++i) {
		std::cout << "Edge " << i << " connecting: "
			<< nodeConnections[2*i] << " and "
			<< nodeConnections[2*i + 1] << '\n';
	}
}


std::shared_ptr<PlatletSystem> PlatletSystemBuilder::Create_Platlet_System_On_Device() {

	// *****************************************************
	// Calculations of parameter values based on vector info (size, max, etc.)




	// *****************************************************
	// Create and initialize (the pointer to) the final system on device.
	 

	// The pointer to the final system to be returned by this method.
	/* std::shared_ptr<PlatletSystem> host_ptr_devPlatletSystem = std::make_shared<PlatletSystem>();

	host_ptr_devPlatletSystem->initializePltSystem();

	return host_ptr_devPlatletSystem; */

	return nullptr;
}




