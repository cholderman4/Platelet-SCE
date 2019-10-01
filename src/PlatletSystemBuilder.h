#ifndef PLATLET_SYSTEM_BUILDER_H_
#define PLATLET_SYSTEM_BUILDER_H_

#include "glm/glm.hpp"
#include <memory>
#include <thrust/host_vector.h>
#include <string>
#include <vector>

#include "AdvancePositionsByForce.h"
#include "AdvancePositionsByVelocity.h"
#include "BucketScheme.h"
#include "FixNodes.h"
#include "MorseForce.h"
// #include "NodeData.h"
#include "ParameterManager.h"
#include "PlatetSystemBuilder.h"
#include "PlatletSystemController.h"
#include "PrintVTK.h"
#include "SpringForce.h"


class ICalculateParameter;
class IFunction;
class IOutputPath;
class ISave;
class NodeData;


/* struct GeneralParams {

    // Parameters used in various formulae. 
    double epsilon{ 0.0001 };
    double dt{ 0.0005 };

	double viscousDamp{ 3.769911184308 }; // ???
	double temperature{ 300.0 };
	double kB{ 1.3806488e-8 };
}; */



class PlatletSystemBuilder {

    private:

    // ***************************************
    // NodeData.
 
    std::vector<glm::dvec3> nodePosition;
    
    thrust::host_vector<double> pos_x;
    thrust::host_vector<double> pos_y;
    thrust::host_vector<double> pos_z;

    std::vector<unsigned> nNodesByType;

    // ***************************************
    // SpringForce

    thrust::host_vector<unsigned> nodeID_L;
    thrust::host_vector<unsigned> nodeID_R;

    thrust::host_vector<double> len_0;

    // ***************************************
    // Outputs


    // ***************************************
    // Functions/Functors

    // Utility functions: e.g. bucket scheme
    std::vector< std::shared_ptr<IFunction> > utilFunctions;

    // Forces
    std::vector< std::shared_ptr<IFunction> > externalForces;
    std::vector< std::shared_ptr<IFunction> > platletForces;

    // File outputs.
    std::vector< std::shared_ptr<ISave> > saveStates;

    std::vector< std::shared_ptr<ICalculateParameter> > parameters;

    std::shared_ptr<IOutputPath> outputPath;
    
    
    // ***************************************
    // This way, the system is initialized immediately (though empty).
    // NodeData will exist and be able to be passed along to objects referencing it.

    // The big daddy!!
    std::unique_ptr<PlatletSystem> platletSystem;

    public:

    PlatletSystemBuilder(std::shared_ptr<IOutputPath> _outputPath);

    // ******************************************
    // Called by director to add individual pieces as they are read from XML.

    int addMembraneNode( glm::dvec3 pos );
    int addInteriorNode( glm::dvec3 pos );
    int addMembraneEdge( unsigned n1, unsigned n2 );
    
    // Probably change this around so that it checks as we go.
    // Or just delete this entirely.
    // void setNodeCount(unsigned count, unsigned nodeTypeA);

    
    void initializeFunctors();

    // Parameter calculations
    ParameterManager paramManager;
    
    // Setting parameter input values.
    void getParameterKeysFromFunctors();
    void setParameterValuesToFunctors();

    // Save parameters and output files.
    void setOutputs();

    std::unique_ptr<PlatletSystem> getSystem();

    // void enrollSaveState(std::shared_ptr<ISave> s);

};


#endif 