#ifndef ADVANCE_POSITIONS_BY_FORCE_H_
#define ADVANCE_POSITIONS_BY_FORCE_H_

class NodeData;

#include "IFunction.h"
#include "NodeTypeOperation.h"


class AdvancePositionsByForce : public NodeTypeOperation, public IFunction {

    private:

    double dt{ 0.0005 };
	double viscousDamp{ 3.769911184308 }; // ???
	double temperature{ 300.0 };
	double kB{ 1.3806488e-8 };

    public: 
        void execute();

        AdvancePositionsByForce(NodeData& _nodeData);
};

#endif