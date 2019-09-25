#ifndef NODE_OPERATION_H_
#define NODE_OPERATION_H_

#include <vector>
#include <memory>

#include <thrust/device_vector.h>

class ParameterManager;
class NodeData;

class NodeOperation {

protected:
    NodeData& nodeData;

    // Begin/End index for each node type.
    thrust::device_vector<unsigned> indicesBegin;
    thrust::device_vector<unsigned> indicesEnd;

    // Used to retrieve/update info for node types.
    // The value will be used to interface with NodeData
    // The index will be used internally by the functor
    std::vector<unsigned> nodeTypes;
    unsigned nNodeTypes;
    
    // Begin/End index for all nodes.
    unsigned indexBegin { 0 };
    unsigned indexEnd { 0 };
        
    // unsigned nRegisteredNodeTypes;
    // unsigned nRegisteredNodes;

    std::vector<unsigned> nodeInteractionA;
    std::vector<unsigned> nodeInteractionB;

    

public:

    // NodeOperation();

    ~NodeOperation();

    NodeOperation(NodeData& _nodeData);

    void registerNodeInteraction(unsigned nodeTypeA, unsigned nodeTypeB);

    void updateNodeIndices();

    void checkBeginEnd(unsigned begin, unsigned end);

    virtual void getDefaultParameterValues(ParameterManager& parameterManager) = 0;
};

#endif