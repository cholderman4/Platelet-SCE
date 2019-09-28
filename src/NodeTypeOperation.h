#ifndef NODE_OPERATION_H_
#define NODE_OPERATION_H_

#include <vector>
#include <memory>

#include <thrust/device_vector.h>

class IReadParameter;
class NodeData;
class ParameterManager;

class NodeTypeOperation {

    protected:
    NodeData& nodeData;

    // Begin/End index for each node type.
    thrust::device_vector<unsigned> indicesBegin;
    thrust::device_vector<unsigned> indicesEnd;

    // Used to retrieve/update info for node types.
    // The value will be used to interface with NodeData
    // The index will be used internally by the functor
    std::vector<unsigned> nodeTypes;
    unsigned nNodeTypes{ 0 };
    
    // Begin/End index for all nodes.
    unsigned indexBegin { 0 };
    unsigned indexEnd { 0 };
        
    // unsigned nRegisteredNodeTypes;
    // unsigned nRegisteredNodes;

    void checkBoundary(unsigned begin, unsigned end);
    

    public:
    NodeTypeOperation(NodeData& _nodeData);
    ~NodeTypeOperation();

    void updateNodeIndices();

    void registerNodeType(unsigned nodeType);

    virtual void getParameterKeys(IParameterList& paramList) = 0;
    virtual void setParameterValues(const IReadParameter& readParam) = 0;
};

#endif // NODE_OPERATION_H_