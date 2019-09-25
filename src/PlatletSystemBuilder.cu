#include "PlatletSystemBuilder.h"

#include "BucketScheme.h"
#include "MorseForce.h"
// #include "NodeData.h"
#include "PlatletSystem.h"
// #include "PlatletSystemController.h"
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

    // *****************************************************
    // Initialize all the functors

    // Utility functors (bucketScheme)
    auto bucketScheme = std::make_shared<BucketScheme>(nodeData, paramManager);
    utilFunctions.push_back(bucketScheme);

    // Platlet Forces
    platletForces.push_back(std::make_shared<SpringForce>(nodeData, paramManager));
    platletForces.push_back(std::make_shared<MorseForce>(nodeData, *bucketScheme, paramManager));
    platletForces.push_back(std::make_shared<AdvancePositionsByForce>(nodeData, paramManager));

    // External forces
    // externalForces.push_back((std::make_shared<PullingForce>(nodeData, paramManager));
    // externalForces.push_back((std::make_shared<FixNodes>(nodeData, paramManager));

  

    
}


void PlatletSystemBuilder::setInputs() {

    // Read parameters from XML.

}


void PlatletSystemBuilder::setOutputs() {
    // Register the node types with functors
    platletBuilder->enrollCalculateParameter(std::make_shared<KineticEnergy>(nodeData));


    // Register save files.
    platletBuilder->enrollSaveState(std::make_shared<PrintVTK>(nodeData))

}