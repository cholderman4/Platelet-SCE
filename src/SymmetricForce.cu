#include "SymmetricForce.h"


SymmetricForce::SymmetricForce(NodeData& _nodeData) :
    NodeTypeOperation(_nodeData) {};


void SymmetricForce::registerNodeInteraction(unsigned nodeTypeA, unsigned nodeTypeB) {

    nodeInteractionA.push_back(nodeTypeA);
    nodeInteractionB.push_back(nodeTypeB);

    registerNodeType(nodeTypeA);
    
    if (nodeTypeA != nodeTypeB) {
        registerNodeType(nodeTypeB);
    }
    
}

