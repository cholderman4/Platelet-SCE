#include "PlatletSystem.h"
#include "PlatletSystemBuilder.h"
#include "PlatletStorage.h"
// #include "SystemStructures.h"


PlatletStorage::PlatletStorage(std::weak_ptr<PlatletSystem> a_pltSystem,
	std::weak_ptr<PlatletSystemBuilder> b_pltSystem, 
	const std::string& a_fileName) {

	pltSystem = a_pltSystem;
	pltBuilder = b_pltSystem;
	fileNameDescription = a_fileName;

};


void PlatletStorage::print_VTK_File() {

	std::shared_ptr<PlatletSystem> pltSys = pltSystem.lock();

	// Save membrane node positions to VTK file.
	if (pltSys) {

		++outputCounter;
		
		unsigned digits = ceil(log10(outputCounter + 1));
		std::string format = ".vtk";
		std::string Number;
		std::string initial = "AnimationTest/PlatletMembrane_";
		std::ofstream ofs;
		if (digits == 1 || digits == 0) {
			Number = "0000" + std::to_string(outputCounter);
		}
		else if (digits == 2) {
			Number = "000" + std::to_string(outputCounter);
		}
		else if (digits == 3) {
			Number = "00" + std::to_string(outputCounter);
		}
		else if (digits == 4) {
			Number = "0" + std::to_string(outputCounter);
		}

		std::string Filename = initial + fileNameDescription + Number + format;

		ofs.open(Filename.c_str());


		unsigned memNodeCount = pltSys->memNode.count;
		//__attribute__ ((unused)) unsigned maxNeighborCount = (pltSys->generalParams).maxNeighborCount;

		unsigned springEdgeCount = pltSys->springEdge.count;

		ofs << "# vtk DataFile Version 3.0" << std::endl;
		ofs << "Point representing Sub_cellular elem model" << std::endl;
		ofs << "ASCII" << std::endl << std::endl;
		ofs << "DATASET UNSTRUCTURED_GRID" << std::endl;


		ofs << "POINTS " << memNodeCount << " float" << std::endl;
		for (unsigned i = 0; i < memNodeCount; i++) { 
			double pos_x = pltSys->memNode.pos_x[i];
			double pos_y = pltSys->memNode.pos_y[i];
			double pos_z = pltSys->memNode.pos_z[i];

			ofs << std::setprecision(5) <<std::fixed<< pos_x << " " << pos_y << " " << pos_z << " " << '\n'<< std::fixed;
		}		
		
		ofs.close();
	}

};
