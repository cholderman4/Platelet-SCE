#ifndef ADVANCE_POSITIONS_BY_VELOCITY_H_
#define ADVANCE_POSITIONS_BY_VELOCITY_H_

#include "IFunction.h"
#include "NodeOperation.h"

class NodeData;



class AdvancePositionsByVelocity : public IFunction, public NodeOperation {

    private:
    double dt;
    double vel_x;
    double vel_y;
    double vel_z;



    public:
    void execute();

    AdvancePositionsByVelocity(NodeData& _nodeData);
    
    void SetVelocity(double _vel_x, double _vel_y, double _vel_z);
    
    void SetDirectionAndMagnitude(double _dir_x, double _dir_y, double _dir_z, double _magnitude);

    void SetTimeStep(double _dt);
};



#endif