
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

std::string generateOutputFileName(std::string inputFileName) {
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

    if (auto p = props.child("absolute-temperature"))
        pltBuilder->defaultTemperature = (p.text().as_double());
        
    // *****************************************************

    // Add membrane nodes.


    // Loop through node siblings and do pltBuilder->addMembraneNode


    // *****************************************************

    // Add membrane spring connections.


    // Loop through link siblings and do pltBuilder->addMembraneEdge().


    // *****************************************************
	// Create and initialize (the pointer to) the final system on device().

    auto pltModel = builder->Create_Platlet_System_On_Device();

	std::cout << "model built" << "\n";

	return pltModel;




    
}


void run() {

    double epsilon = 0.01;
    double timeStep = 0.001;

    // Inital creation of pointer to the PlatletSystemBuilder.
    auto pltBuilder = std::make_shared<PlatletSystemBuilder>(epsilon, timeStep);

    auto pltSystem = createPlatletSystem("testData.xml", pltBuilder);



}


int main() {  

    auto pltSystem = createPlatletSystem();

    PlatletSystem platletSystem;

    auto outputFileName = generateOutputFileName("test");

    platletSystem.initializePltSystem(5,5);

    platletSystem.solvePltSystem();
    
    /* // Test values
    unsigned N{3};
    unsigned E{3};
    Node node(N);
    Edge edge(E);

    // Test initialization of nodes and edges.
    node.printPoints();
    
    edge.printConnections();

    Spring_Force(node, edge); */
        
    return 0;
}