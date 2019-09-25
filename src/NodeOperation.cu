#include "NodeOperation.h"

#include "NodeData.h"

// NodeOperation::NodeOperation() {};

NodeOperation::~NodeOperation() {};

NodeOperation::NodeOperation(NodeData& _nodeData) :
    nodeData(_nodeData) {}

void NodeOperation::registerNodeInteraction(unsigned nodeTypeA, unsigned nodeTypeB) {
    nodeInteractionA.push_back(nodeTypeA);
    nodeInteractionB.push_back(nodeTypeB);

    unsigned begin = nodeData.getIndexBegin(nodeType) ;
    unsigned end = nodeData.getIndexEnd(nodeType) ;

    indicesBegin.push_back(begin);
    indicesEnd.push_back(end);

    checkBeginEnd(begin, end);
}

void NodeOperation::updateNodeIndices() {

    for (auto i = 0; i < nodeTypes.size(); ++i) {
        indicesBegin[i] = nodeData.getIndexBegin(nodeTypes[i]);
        indicesEnd[i] = nodeData.getIndexEnd(nodeTypes[i]);

        checkBeginEnd(indicesBegin[i], indicesEnd[i]);
    }
}

void NodeOperation::checkBeginEnd(unsigned begin, unsigned end) {
    if (begin < indexBegin) {
        indexBegin = begin;
    }
    if (end > indexEnd) { 
        indexEnd = end;
    }
}