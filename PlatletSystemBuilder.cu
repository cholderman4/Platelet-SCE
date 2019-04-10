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


PlatletSystemBuilder::SystemBuilder(double _epsilon, double _dt):
	epsilon(_epsilon), dt(_dt) {}

PlatletSystemBuilder::~SystemBuilder() {
}

unsigned PlatletSystemBuilder::addMembraneNode() {

}


unsigned PlatletSystemBuilder::addMembraneEdge() {

}


std::shared_ptr<PlatletSystem> PlatletSystemBuilder::Create_Platlet_System_On_Device() {

	// *****************************************************
	// Calculations of parameter values based on vector info (size, max, etc.)




	// *****************************************************
	// Create and initialize (the pointer to) the final system on device.
	 

	// The pointer to the final system to be returned by this method.
	std::shared_ptr<PlatletSystem> host_ptr_devPlatletSystem = std::make_shared<PlatletSystem>();

	host_ptr_devPlatletSystem->initializePltSystem(
		node,
		springEdge,
		generalParams);

	return host_ptr_devPlatletSystem;
}




