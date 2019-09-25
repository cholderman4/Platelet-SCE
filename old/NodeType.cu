#include "NodeType.h"

#include "NodeData.h"


void NodeType::updateIndex() {
    indexBegin = nodeData->getIndexBegin(myTypeID);
    indexEnd = nodeData->getIndexEnd(myTypeID);
}