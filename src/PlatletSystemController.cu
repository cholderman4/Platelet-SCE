#include "PlatletSystemController.h"


#include "IFunction.h"
#include "ICalculateParameter.h"
#include "IPrint.h"

PlatletSystemController::PlatletSystemController() {};


void PlatletSystemController::runSystem() {

    // Call initial behavior for saved states.
    for (auto s : saveState) {
        s->beginSimulation();
    }

    // Ensure utility functions have an initial setting.
    for (auto uf : utilFunctions) {
        uf->execute();
    }

    while (checkRunSystem()) {
        // AdvanceTime step
        ++nIterations;
        currentTime += dt;


        // Check for bucket scheme reset.
        if(checkUtilFunctionReset()) {
            for (auto uf : utilFunctions) {
                uf->execute();
            }
        }

        // Apply external forces.
        // These are only here to ensure they happen first.
        for (auto ef : externalForces) {
            ef->execute();
        }

        // Apply platelet forces and advance positions.
        for (auto pf : platletForces) {
            pf->execute();
        }


        // Perhaps there is a better way to ensure parameters are calculated before files are printed.
        // Calculate parameter values.
        if (checkParameterCalculation()) {
            for (auto p : parameters) {
                p->calculate();
            }
        }

        // Check file outputs.
        if (checkPrintFile()) {
            for (auto s : saveState) {
                s->save();
            }
        }
    }

    // Calculate parameters one last time
    for (auto p : parameters) {
        p->calculate();
    }
    
    // Call final behavior for saved states.
    for (auto s : saveState) {
        s->endSimulation();
    }
}


bool PlatletSystemController::checkRunSystem() {
    if (nIterations == maxIterations) {
        return false;
    } else {
        return true;
    } 
}


bool PlatletSystemController::checkUtilFunctionReset() {
    if (nIterations % resetUtilFunctionStepSize == 0) {
        return true;
    } else {
        return false;
    }
}


bool PlatletSystemController::checkPrintFile() {
    if (nIterations % printFileStepSize == 0) {
        return true;
    } else {
        return false;
    }
}


bool PlatletSystemController::checkParameterCalculation() {
    // Perhaps there is a better way to ensure parameters are calculated before files are printed.
    return checkPrintFile();
}


void PlatletSystemController::enrollUtilFunction(std::shared_ptr<IFunction> f) {
    utilFunctions.push_back(f);
}

void PlatletSystemController::enrollExternalForce(std::shared_ptr<IFunction> f) {
    externalForces.push_back(f);
}

void PlatletSystemController::enrollPlatletForce(std::shared_ptr<IFunction> f) {
    platletForces.push_back(f);
}

void PlatletSystemController::enrollPrintFile(std::shared_ptr<ISave> s) {
    saveStates.push_back(s);
}

void PlatletSystemController::enrollCalculateParameter(std::shared_ptr<ICalculateParameter> p) {
    parameters.push_back(p);
}