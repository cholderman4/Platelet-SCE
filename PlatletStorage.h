#ifndef PLATLET_STORAGE_H_
#define PLATLET_STORAGE_H_

#include <fstream>	     
#include <memory>
#include <iomanip>

class PlatletSystem;
class PlatletSystemBuilder;

//During graph deformation, this file stores position and velocity of nodes at a given time step
class PlatletStorage {
	
	std::weak_ptr<PlatletSystem> pltSystem;
	std::weak_ptr<PlatletSystemBuilder> pltBuilder;
	std::ofstream output;
	std::ofstream statesOutput;
	
	std::ofstream statesOutputStrain;
	std::string bn;

	unsigned stepCounter = 0;
	unsigned stepInterval = 10;

	double forceStep = 0.0;	 //increments in which force will be increased. 
	double magnitudeForce = 0.0;  //how much force we are currently at.
	/* int currentAddedEdges = 0;
	int previousAddedEdges = 0; */
	unsigned outputCounter = 0;

public: 
	PlatletStorage(std::weak_ptr<PlatletSystem> a_system,
		std::weak_ptr<PlatletSystemBuilder> b_system, 
		const std::string& a_filename);

	void save_params(void);

	void updateStorage(void);

	void print_VTK_File(void);
};
#endif