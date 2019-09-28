#include "NodeTypeOperation.h"

#include "NodeData.h"


NodeTypeOperation::~NodeTypeOperation() {};

NodeTypeOperation::NodeTypeOperation(NodeData& _nodeData) :
    nodeData(_nodeData) {}

void NodeTypeOperation::registerNodeType(unsigned nodeType) {
    // Search for duplicates before adding to vector.
    if ( std::find(nodeTypes.begin(), nodeTypes.end(), nodeType) == vec.end() ) {
        nodeTypes.push_back(nodeType);
        ++nNodeTypes;

        unsigned begin = nodeData.getIndexBegin(nodeType) ;
        unsigned end = nodeData.getIndexEnd(nodeType) ;

        indicesBegin.push_back(begin);
        indicesEnd.push_back(end);

        checkBoundary(begin, end);
    }    
}

void NodeTypeOperation::updateNodeIndices() {

    for (auto i = 0; i < nodeTypes.size(); ++i) {
        indicesBegin[i] = nodeData.getIndexBegin(nodeTypes[i]);
        indicesEnd[i] = nodeData.getIndexEnd(nodeTypes[i]);

        checkBoundary(indicesBegin[i], indicesEnd[i]);
    }
}

void NodeTypeOperation::checkBoundary(unsigned begin, unsigned end) {
    if (begin < indexBegin) {
        indexBegin = begin;
    }
    if (end > indexEnd) { 
        indexEnd = end;
    }
}