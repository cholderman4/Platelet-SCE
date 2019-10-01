#ifndef FIX_NODES_H_
#define FIX_NODES_H_


#include <thrust/device_vector.h>


#include "IFunction.h"
#include "NodeOperation.h"


class FixNodes : public NodeOperation, public IFunction {


    public:
    FixNodes(NodeData& _nodeData);

    void execute();
};

#endif