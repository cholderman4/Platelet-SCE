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


		unsigned memNodeCount = pltSys->node.membrane_count;
		unsigned intNodeCount = pltSys->node.interior_count;
		unsigned total_node_count = pltSys->node.total_count;
		//__attribute__ ((unused)) unsigned maxNeighborCount = (pltSys->generalParams).maxNeighborCount;

		unsigned springEdgeCount = pltSys->springEdge.count;

		ofs << "# vtk DataFile Version 3.0" << std::endl;
		ofs << "Point representing Sub_cellular elem model" << std::endl;
		ofs << "ASCII" << std::endl;
		ofs << "DATASET POLYDATA" << std::endl;


		ofs << "POINTS " << total_node_count << " FLOAT" << std::endl;
		for (unsigned i = 0; i < total_node_count; ++i) { 
			double pos_x = pltSys->node.pos_x[i];
			double pos_y = pltSys->node.pos_y[i];
			double pos_z = pltSys->node.pos_z[i];

			ofs << std::setprecision(5) <<std::fixed<< pos_x << " " << pos_y << " " << pos_z << " " << '\n' << std::fixed;
		}

		/* for (unsigned i = 0; i < intNodeCount; ++i) { 
			double pos_x = pltSys->intNode.pos_x[i];
			double pos_y = pltSys->intNode.pos_y[i];
			double pos_z = pltSys->intNode.pos_z[i];

			ofs << std::setprecision(5) <<std::fixed<< pos_x << " " << pos_y << " " << pos_z << " " << '\n' << std::fixed;
		} */


		// Print info for Membrane vs Internal node.
		/* ofs << "POINT_DATA " << memNodeCount + intNodeCount << std::endl;
		ofs << "SCALARS IsMembraneNode FLOAT \n";
		ofs << "LOOKUP_TABLE default \n";
		for (unsigned i = 0; i < memNodeCount; ++i) { 
			ofs << "1.0 \n";
		}

		for (unsigned i = 0; i < intNodeCount; ++i) { 
			ofs << "0.0 \n";
		} */
		ofs.close();
	}

};
