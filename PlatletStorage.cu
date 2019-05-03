#include "PlatletSystem.h"
#include "PlatletSystemBuilder.h"
#include "PlatletStorage.h"
// #include "SystemStructures.h"


PlatletStorage::PlatletStorage(std::weak_ptr<PlatletSystem> a_pltSystem,
	std::weak_ptr<PlatletSystemBuilder> b_pltSystem, 
	__attribute__ ((unused)) const std::string& a_fileName) {

	pltSystem = a_pltSystem;
	pltBuilder = b_pltSystem;

};



/* void Storage::save_params(void) {
	std::shared_ptr<System>pltSys =pltSystem.lock();
	if (pltSys) {

		//first create a new file using the current network strain
		
		std::string format = ".sta";
		
		std::string strain =  std::to_string(pltSys->generalParams.currentTime);
		std::string initial = "Params/Param_";
		std::ofstream ofs;
		std::string Filename = initial + strain + format;
		ofs.open(Filename.c_str());



		//unsigned maxNeighborCount =pltSys->generalParams.maxNeighborCount;
		unsigned memNodeCount =pltSys->generalParams.memNodeCount;
		unsigned originalNodeCount =pltSys->generalParams.originNodeCount;
		unsigned originalEdgeCount =pltSys->generalParams.originLinkCount;
		unsigned edgeCountDiscretize =pltSys->generalParams.originEdgeCount;
		//Now first place strain
		ofs << std::setprecision(5) <<std::fixed<< "time " <<pltSys->generalParams.currentTime<<std::endl;
		ofs << std::setprecision(5) <<std::fixed<< "minX " <<pltSys->domainParams.minX<<std::endl;
		ofs << std::setprecision(5) <<std::fixed<< "maxX " <<pltSys->domainParams.maxX<<std::endl;
		ofs << std::setprecision(5) <<std::fixed<< "minY " <<pltSys->domainParams.minY<<std::endl;
		ofs << std::setprecision(5) <<std::fixed<< "maxY " <<pltSys->domainParams.maxY<<std::endl;
		ofs << std::setprecision(5) <<std::fixed<< "minZ " <<pltSys->domainParams.minX<<std::endl;
		ofs << std::setprecision(5) <<std::fixed<< "maxZ " <<pltSys->domainParams.maxX<<std::endl;
		
		
		ofs << std::setprecision(5) <<std::fixed<< "original_node_count " << originalNodeCount <<std::endl;
		ofs << std::setprecision(5) <<std::fixed<< "node_count_discretize " << memNodeCount <<std::endl;
		ofs << std::setprecision(5) <<std::fixed<< "original_edge_count " << originalEdgeCount <<std::endl;
		ofs << std::setprecision(5) <<std::fixed<< "edge_count_discretize " << edgeCountDiscretize <<std::endl;
		
		//place nodes
		for (unsigned i = 0; i <pltSys->node.nodeLocX.size(); i++) {
			double x =pltSys->node.nodeLocX[i];
			double y =pltSys->node.nodeLocY[i];
			double z =pltSys->node.nodeLocZ[i];
			ofs << std::setprecision(5) <<std::fixed<< "node " << x << " " << y << " " << z <<std::endl;
		
		}
		
		//place plts
		for (unsigned i = 0; i <pltSys->pltInfoVecs.pltLocX.size(); i++) {
			double x =pltSys->pltInfoVecs.pltLocX[i];
			double y =pltSys->pltInfoVecs.pltLocY[i];
			double z =pltSys->pltInfoVecs.pltLocZ[i];
			ofs << std::setprecision(5) <<std::fixed<< "plt " << x << " " << y << " " << z <<std::endl;
		
		}
		//place force node is experiencing
		for (unsigned i = 0; i <pltSys->node.nodeLocX.size(); i++) {
			ofs << std::setprecision(5) <<std::fixed<< "force_on_node " <<pltSys->node.sumForcesOnNode[i]<<std::endl;
		
		}

		//place original edges
		for (unsigned edge = 0; edge <pltSys->generalParams.originEdgeCount; edge++) {
			unsigned idL =pltSys->node.deviceEdgeLeft[edge];
			unsigned idR =pltSys->node.deviceEdgeRight[edge];
			ofs <<"original_edge_discretized " <<idL <<" "<< idR <<std::endl;
			
		}
				 
		//place added edges
		for (unsigned edge =pltSys->generalParams.originEdgeCount; edge <pltSys->generalParams. springEdgeCount; edge++) {
			unsigned idL =pltSys->node.deviceEdgeLeft[edge];
			unsigned idR =pltSys->node.deviceEdgeRight[edge];
			ofs <<"added_edge " <<idL <<" "<< idR <<std::endl;
			
		}

		//original edge strain
		for (unsigned i = 0; i <pltSys->generalParams.originEdgeCount; i++ ){
			double val =pltSys->node.discretizedEdgeStrain[i];

			ofs << std::setprecision(5)<< std::fixed<<"original_edge_strain " << val <<std::endl;
		}
				
		//original edge alignment
		for (unsigned i = 0; i <pltSys->generalParams.originEdgeCount; i++ ){
			double val =pltSys->node.discretizedEdgeAlignment[i];
			ofs << std::setprecision(5)<< std::fixed<<"original_edge_alignment " << val <<std::endl;
		}

		//added edge strain
		for (unsigned i =pltSys->generalParams.originEdgeCount; i <pltSys->generalParams. springEdgeCount; i++ ){
			double val =pltSys->node.discretizedEdgeStrain[i];
			ofs << std::setprecision(5)<< std::fixed<<"added_edge_strain " << val <<std::endl;
		}
		
		//added links per node.
		for (unsigned i = 0; i <pltSys->generalParams.memNodeCount; i++ ){
			unsigned val =pltSys->wlcInfoVecs.currentNodeEdgeCountVector[i] - 
				pltSys->wlcInfoVecs.numOriginalNeighborsNodeVector[i];
			ofs << std::setprecision(5)<< std::fixed<<"bind_sites_per_node " << val <<std::endl;
		}



	}
}; */



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

		std::string Filename = initial + Number + format;

		ofs.open(Filename.c_str());


		unsigned memNodeCount = pltSys->generalParams.memNodeCount;
		//__attribute__ ((unused)) unsigned maxNeighborCount = (pltSys->generalParams).maxNeighborCount;

		unsigned springEdgeCount = pltSys->generalParams.springEdgeCount;

		ofs << "# vtk DataFile Version 3.0" << std::endl;
		ofs << "Point representing Sub_cellular elem model" << std::endl;
		ofs << "ASCII" << std::endl << std::endl;
		ofs << "DATASET UNSTRUCTURED_GRID" << std::endl;


		ofs << "POINTS " << memNodeCount << " float" << std::endl;
		for (unsigned i = 0; i < memNodeCount; i++) { 
			double pos_x = pltSys->node.pos_x[i];
			double pos_y = pltSys->node.pos_y[i];
			double pos_z = pltSys->node.pos_z[i];

			ofs << std::setprecision(5) <<std::fixed<< pos_x << " " << pos_y << " " << pos_z << " " << '\n'<< std::fixed;
		}		
		
		ofs.close();
	}

};
