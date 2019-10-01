#include "PlatletSystemBuilder.h"

#include "BucketScheme.h"
#include "FixNodes.h"
#include "KineticEnergy.h"
#include "MorseForce.h"
// #include "NodeData.h"
// #include "NodeOperation.h"
#include "PlatletSystem.h"
// #include "PlatletSystemController.h"
#include "PrintVTK.h"
#include "PrintNodePositionsVTK.h"
#include "PrintNodeTypeVTK.h"
#include "SetNodeSlice.h"
#include "SpringForce.h"



PlatletSystemBuilder::PlatletSystemBuilder(
    std::shared_ptr<IOutputPath> _outputPath) :
    platletSystem(std::make_unique<PlatletSystem>())
    outputPath(std::move(_outputPath)) {};


int PlatletSystemBuilder::addMembraneNode( glm::dvec3 pos ) {
// Get from old builder
}


int PlatletSystemBuilder::addInteriorNode( glm::dvec3 pos ) {
// Get from old builder
}


int PlatletSystemBuilder::addMembraneEdge( unsigned n1, unsigned n2 ) {
// Get from old builder
}


void PlatletSystemBuilder::initializeFunctors() {

    // Utility functors (bucketScheme)
    auto bucketScheme = std::make_shared<BucketScheme>(platletSystem->nodeData, paramManager);
    utilFunctions.push_back(bucketScheme);

    // Platlet Forces
    auto springForce = std::make_shared<SpringForce>(platletSystem->nodeData);
    springForce->registerNodeInteraction(0,0); // membrane-membrane
    platletForces.push_back(springForce);

    auto morseForce = std::make_shared<MorseForce>(platletSystem->nodeData, *bucketScheme);
    morseForce->registerNodeInteraction(0,1) // membrane-interior
    morseForce->registerNodeInteraction(1,0) // interior-membrane
    morseForce->registerNodeInteraction(1,1) // interior-interior
    platletForces.push_back(morseForce);

    // Must be last.
    platletForces.push_back(std::make_shared<AdvancePositionsByForce>(platletSystem->nodeData));

    // External forces
    unsigned nMembraneNodes = nNodesByType[0];
    auto setNodeSlice =  SetNodeSlice(
        pos_z.begin(), 
        pos_z.begin() + nMembraneNodes,
        0.10,
        true);

    /* auto pullingForce = std::make_shared<PullingForce>(platletSystem->nodeData);
    pullingForce->getNodeID(setNodeSlice);
    externalForces.push_back(pullingForce); */

    auto fixNodes = std::make_shared<FixNodes>(platletSystem->nodeData);
    setNodeSlice.setIsTop(false);
    fixNodes->getNodeID(setNodeSlice);
    externalForces.push_back(fixNodes);
}


void PlatletSystemBuilder::getParameterKeysFromFunctors() {
    // Get parameter key requests from functors.
    for (auto ef : externalForces) {
        ef->getParameterKeys(paramManager);
    }

    for (auto ef : platletForces) {
        pf->getParameterKeys(paramManager);
    }
}

// Director is responsible for filling in parameter values in between these function calls.

void PlatletSystemBuilder::setParameterValuesToFunctors() {
    // Send parameter values back to functors.
    // Director should modify parameter values before this step.
    // If not, then we are just using default values.
    for (auto ef : externalForces) {
        ef->setParameterValues(paramManager);
    }

    for (auto ef : platletForces) {
        pf->setParameterValues(paramManager);
    }
}



void PlatletSystemBuilder::setOutputs() {

    // Set parameter outputs.
    // Used to print and/or for testing as simulation is running.
    auto kineticEnergy = std::make_shared<KineticEnergy>(platletSystem->nodeData);
    
    parameters.push_back(kineticEnergy);


    // Register save files.
    auto printVTK = std::make_shared<PrintVTK>(platletSystem->nodeData)
    printVTK->enrollData(std::make_shared<PrintNodePositionsVTK>(platletSystem->nodeData));
    printVTK->enrollData(std::make_shared<PrintNodeTypeVTK>(platletSystem->nodeData));
    // Set OutputPath!!
    saveStates.push_back(printVTK);


    // saveStates.push_back(printKineticEnergy);
    // saveStates.push_back(printAllParameters);

}