
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

#include "PlatletSystem.h"

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

// hard coded values for now. Later will accept xml file for intial values.
std::shared_ptr<PlatletSystem> createPlatletSystem() {

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