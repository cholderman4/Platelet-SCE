
// Not sure how many of these are actually needed.
// ************************************************
#include <iomanip>
#include <string>
#include <memory>
#include <fstream>
#include <ctime>
#include <stdio.h>
#include <inttypes.h>
#include <cstddef>
// ************************************************

#include "pugixml/include/pugixml.hpp"

#include "PlatletSystem.h"
#include "PlatletSystemBuilder.h"
#include "PlatletStorage.h"

std::string generateOutputFileNameByDate(std::string inputFileName) {
	time_t now;
	const int MAX_DATE = 64;
	char theDate[MAX_DATE];
	theDate[0] = '\0';

	now = time(nullptr);

	if (now != -1) {
		strftime(theDate, MAX_DATE, "_%Y.%m.%d_%H-%M-%S", gmtime(&now));
		return inputFileName + theDate;
	}
	return "";
}


std::shared_ptr<PlatletSystem> createPlatletSystem(const char* schemeFile, std::shared_ptr<PlatletSystemBuilder> pltBuilder) {
    // INPUT:   xml file with inital data
    //          pointer to PlatletSystemBuilder
    // OUTPUT:  pointer to the PlatletSystem (on device) with values read in through PlatletSystemBuilder.


    pugi::xml_document doc;
	pugi::xml_parse_result parseResult = doc.load_file(schemeFile);

	if (!parseResult) {
		std::cout << "parse error in createPlatletSystem: " << parseResult.description() << std::endl;
		return nullptr;
	}
	pugi::xml_node root = doc.child("data");
	pugi::xml_node nodes = root.child("nodes");
	pugi::xml_node links = root.child("links");
	pugi::xml_node props = root.child("settings");

    // Check that no crucial data is missing.
    // Default parameter values are stored in GeneralParams, so those are optional.
	if (!(root && nodes && links)) {
		std::cout << "Unable to find nessesary data.\n";
    }
    
    // *****************************************************
    // Load properties from the "settings" section.
    // Check for parameters that match with GeneralParams (?)

    /* if (auto p = props.child("absolute-temperature"))
        pltBuilder->defaultTemperature = (p.text().as_double());
         */
    // *****************************************************

    // Add membrane nodes.

    double x, y, z; //variables to be used reading in data.

	for (auto node = nodes.child("node"); node; node = node.next_sibling("node")) {
		const char* text = node.text().as_string();

		if (3 != sscanf(text, "%lf %lf %lf", &x, &y, &z)) {
			std::cout << "parse node error\n";
			return 0;
		}
		__attribute__ ((unused)) int unused = pltBuilder->addMembraneNode( glm::dvec3(x, y, z) );
    }
    
    // Check that the nodes are all there.
    // pltBuilder->printNodes();

    // *****************************************************

    // Add membrane spring connections.

    double n1, n2; //variables to be used reading in data.

	for (auto link = links.child("link"); link; link = link.next_sibling("link")) {
		const char* text = link.text().as_string();

		if (2 != sscanf(text, "%lf %lf", &n1, &n2)) {
			std::cout << "parse node error\n";
			return 0;
		}
		__attribute__ ((unused)) int unused = pltBuilder->addMembraneEdge( n1, n2 );
    }

    pugi::xml_node fixedRoot = root.child("fixed");
	if (fixedRoot) {
		for (auto node = fixedRoot.child("node"); node; node = node.next_sibling("node"))
			pltBuilder->fixNode(node.text().as_uint());
	}

    // pltBuilder->printEdges();
    // *****************************************************
	// Create and initialize (the pointer to) the final system on device().

    std::cout << "Building model...\n";
    
    auto ptr_Platlet_System_Host = pltBuilder->Create_Platlet_System_On_Device();

	std::cout << "model built" << "\n";

    return ptr_Platlet_System_Host;
    
    //return nullptr;




    
}


void run() {

    // Used to calculate the time to run the entire system.
    time_t t0,t1;
	t0 = time(0);

    double epsilon = 0.01;
    double timeStep = 0.01;

    // Inital creation of pointer to the PlatletSystemBuilder.
    auto pltBuilder = std::make_shared<PlatletSystemBuilder>(epsilon, timeStep);

    auto pltSystem = createPlatletSystem("info.xml", pltBuilder);

    auto outputFileName = generateOutputFileNameByDate("Test");

    std::cout << outputFileName << '\n';

    auto pltStorage = std::make_shared<PlatletStorage>(pltSystem, pltBuilder, outputFileName);

    pltSystem->assignPltStorage(pltStorage);

    pltSystem->solvePltSystem();

    // ***********************************************************
    // Output the time it takes to run the simulation.
    t1 = time(0);  //current time at the end of solving the system.
	int total,hours,min,sec;
	total = difftime(t1,t0);
	hours = total / 3600;
	min = (total % 3600) / 60;
	sec = (total % 3600) % 60;
	std::cout << "Total time hh: " << hours << " mm: " << min << " ss: " << sec << '\n';
    // ***********************************************************
}


int main(int argc, char** argv) {  

    // run();
           
    return 0;
}