#include "NodeOperation.h"

#include "INodePredicate.h"


NodeOperation::NodeOperation(NodeData& _nodeData) :
    nodeData(_nodeData) {};


NodeOperation::~NodeOperation() {};


void NodeOperation::setNodeID(INodePredicate& nodePredicate) {
    nodePredicate.getNodeID(nodeID);
}
