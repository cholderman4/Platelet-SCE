#include <cassert>
#include <ctime>
#include <memory>
#include <string>
#include <ostream>

#include "glm/glm.hpp"

#include "BucketScheme.h"
#include "Director.h"
#include "MorseForce.h"
#include "NodeData.h"
#include "ParameterManager.h"
#include "PlatetSystemBuilder.h"
#include "PlatletSystemController.h"
#include "SpringForce.h"





set_param_data(
    std::shared_ptr<PlatletSystemBuilder> platletBuilder, 
    std::string paramDataFile,
    NodeData& nodeData, 
    std::string outputPath, 
    int argc, char** argv) {

    // *****************************************************
    // Initialize all the functors

    // Utility functors (bucketScheme)
    auto bucketScheme = std::make_shared<BucketScheme>(nodeData);
    platletBuilder->enrollUtilFunction(bucketScheme);

    // Platlet Forces
    platletBuilder->enrollPlatletForce(std::make_shared<SpringForce>(nodeData));
    platletBuilder->enrollPlatletForce(nodeData, *bucketScheme);
    platletBuilder->enrollPlatletForce(std::make_shared<AdvancePositionsByForce>(nodeData));

    // External forces
    // platletBuilder->enrollExternalForce(std::make_shared<PullingForce>(nodeData));
    // platletBuilder->enrollExternalForce(std::make_shared<FixNodes>(nodeData));

    // Parameter calculation


    

    // Read parameters from XML.
    

    // *****************************************************
    // Load properties from the "parameters" section.
    // Check for parameters that match with parameters logged by the functions.

    

}



void runPlatletTest(int argc, char** argv) {
    // Used to calculate the time to run the entire system.
    time_t t0,t1;
	t0 = time(0);

    // Set filename convention.
    auto fileByDate = std::make_shared<OutputPathByDate>(); 

    // Make PlatletBuilder
    auto platletBuilder = std::make_unique<PlatletSystemBuilder>(fileByDate);

    // Initialize Builder with nodeData and other unchanging values.
    std::string initialDataFile = "NodeInfo.xml";

    // Initialize any parameter savers that persist across multiple simulations.

    unsigned nSimulationRuns = 1;

    for (auto i=0; i < nSimulationRuns; ++i) {

        // Later change this to append numbers corrensponding to repeated tests.
        std::string paramDataFile = "ParamInfo.xml";

        // Maybe move this to a separate function so that it gets deleted out of scope.
        Director director(
            std::move(platletBuilder),
            initialDataFile, 
            paramDataFile);

        director.createPlatletSystem();
        
        // Create PlatletSystem
        auto platletSystem = director->getPlatletSystem();

        // Run platlet system
        platletSystem->runSystem();
    }

    t1 = time(0);  //current time at the end of solving the system.
    int total,hours,min,sec;
    total = difftime(t1,t0);
    hours = total / 3600;
    min = (total % 3600) / 60;
    sec = (total % 3600) % 60;
    std::cout   << "Total time hh: " << hours 
                << " mm: " << min 
                << " ss: " << sec << '\n';
}