#ifndef NODE_OPERATION_H_
#define NODE_OPERATION_H_

#include <thrust/device_vector.h>

class INodePredicate;
class NodeData;

class NodeOperation {

    protected:
    NodeData& nodeData;

    thrust::device_vector<unsigned> nodeID;   
    

    public:
    NodeOperation(NodeData& _nodeData);
    ~NodeOperation();

    void setNodeID(INodePredicate& nodePredicate);

    // void updateNodeIndices();


};

#endif // NODE_OPERATION_H_