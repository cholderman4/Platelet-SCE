#ifndef PLATLET_SYSTEM_CONTROLLER_H_
#define PLATLET_SYSTEM_CONTROLLER_H_

#include <memory>
#include <vector>


class IFunction;
class ICalculateParameter;
class ISave;


class PlatletSystemController {

    private:

    // Utility functions: e.g. bucket scheme
    std::vector< std::shared_ptr<IFunction> > utilFunctions;

    // Forces
    std::vector< std::shared_ptr<IFunction> > externalForces;
    std::vector< std::shared_ptr<IFunction> > platletForces;

    // File outputs.
    std::vector< std::shared_ptr<ISave> > saveStates;


    // Parameter calculations
    std::vector< std::shared_ptr<ICalculateParameter> > parameters;
    
    
    double currentTime{ 0.0 };
    double dt{ 0.0005 };

    
    // Simulation helper functions/parameters.
    unsigned nIterations{ 0 };
    unsigned maxIterations{ 40000 };
    bool checkRunSystem();


    // Bucket scheme resets every 
    unsigned resetUtilFunctionStepSize{ 10 };
    bool checkUtilFunctionReset();


    unsigned printFileStepSize{ 500 };
    bool checkPrintFile();


    bool checkParameterCalculation();





    public:

    PlatletSystemController();

    void runSystem();

};


#endif // PLATLET_SYSTEM_CONTROLLER_H_