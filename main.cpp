
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

    std::cout << "Setting pugixml variables\n";
	pugi::xml_node root = doc.child("data");
	pugi::xml_node memNodes = root.child("membrane-nodes");
    pugi::xml_node intNodes = root.child("interior-nodes");
	pugi::xml_node links = root.child("links");
	pugi::xml_node props = root.child("settings");

    // Check that no crucial data is missing.
    // Default parameter values are stored in GeneralParams, so those are optional.
	if (!(root && memNodes && links)) {
		std::cout << "Unable to find nessesary data.\n";
    }
    
    // *****************************************************
    // Load properties from the "settings" section.
    // Check for parameters that match with GeneralParams (?)

    std::cout << "Loading properties from xml\n";

    if (auto p = props.child("temperature"))
        pltBuilder->generalParams.temperature = (p.text().as_double());
        
    if (auto p = props.child("viscousDamp"))
        pltBuilder->generalParams.viscousDamp = (p.text().as_double());

    if (auto p = props.child("kB"))
        pltBuilder->generalParams.kB = (p.text().as_double());
            
    if (auto p = props.child("memSpringStiffness"))
        pltBuilder->memSpringStiffness = (p.text().as_double());
    
    if (auto p = props.child("memNodeMass"))
        pltBuilder->memNodeMass = (p.text().as_double());

    if (auto p = props.child("intNodeMass"))
        pltBuilder->intNodeMass = (p.text().as_double());

    if (auto p = props.child("memNodeCount"))
        pltBuilder->memNodeCount = (p.text().as_uint());

    if (auto p = props.child("intNodeCount"))
        pltBuilder->intNodeCount = (p.text().as_uint());


     
    // *****************************************************

    std::cout << "Adding membrane nodes\n";
    // Add membrane nodes.
    double x, y, z; //variables to be used reading in data.

    unsigned counter{ 0 };

	for (auto memNode = memNodes.child("mem-node"); memNode; memNode = memNode.next_sibling("mem-node")) {
		const char* text = memNode.text().as_string();
        ++counter;
		if (3 != sscanf(text, "%lf %lf %lf", &x, &y, &z)) {
			std::cout << "parse memNode error\n";
			return 0;
		}
		__attribute__ ((unused)) int unused = pltBuilder->addMembraneNode( glm::dvec3(x, y, z) );
    }

    std::cout << "Counter: " << counter << '\n';



    std::cout << "Adding interior nodes\n";
    // Add interior nodes.
    for (auto intNode = intNodes.child("int-node"); intNode; intNode = intNode.next_sibling("int-node")) {
		const char* text = intNode.text().as_string();

		if (3 != sscanf(text, "%lf %lf %lf", &x, &y, &z)) {
			std::cout << "parse intNode error\n";
			return 0;
		}
		__attribute__ ((unused)) int unused = pltBuilder->addInteriorNode( glm::dvec3(x, y, z) );
    }
    
    // Check that the nodes are all there.
    std::cout << "Printing nodes\n";
    pltBuilder->printNodes();

    // *****************************************************


    std::cout << "Adding membrane spring connections\n";
    // Add membrane spring connections.

    unsigned n1, n2; //variables to be used reading in data.

	for (auto link = links.child("link"); link; link = link.next_sibling("link")) {
		const char* text = link.text().as_string();

		if (2 != sscanf(text, "%u %u", &n1, &n2)) {
			std::cout << "parse link error\n";
			return 0;
		}
        std::cout << "n1: " << n1 << "\t n2: " << n2 << '\n';
		__attribute__ ((unused)) int unused = pltBuilder->addMembraneEdge( n1, n2 );
    }

    std::cout << "Setting fixed nodes\n";
    // Read in fixed nodes.
    pugi::xml_node fixedRoot = root.child("fixed");
	if (fixedRoot) {
		for (auto node = fixedRoot.child("nodeID"); node; node = node.next_sibling("nodeID"))
			pltBuilder->fixNode(node.text().as_uint());
	}

    std::cout << "Printing membrane spring edges\n";
    pltBuilder->printEdges();

    // *****************************************************
	// Create and initialize (the pointer to) the final system on device().

    std::cout << "Building model...\n";
    
    auto ptr_Platlet_System_Host = pltBuilder->Create_Platlet_System_On_Device();

	std::cout << "model built" << "\n";

    return ptr_Platlet_System_Host;    
}


void run() {

    // Used to calculate the time to run the entire system.
    time_t t0,t1;
	t0 = time(0);

    double epsilon = 0.01;
    double timeStep = 0.001;

    // Inital creation of pointer to the PlatletSystemBuilder.
    std::cout << "Creating pltBuilder\n";
    auto pltBuilder = std::make_shared<PlatletSystemBuilder>(epsilon, timeStep);

    std::cout << "Creating pltSystem\n";
    auto pltSystem = createPlatletSystem("info.xml", pltBuilder);

    std::cout << "Generating output filename\n";
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

    std::cout << "Running\n";
    run();
           
    return 0;
}