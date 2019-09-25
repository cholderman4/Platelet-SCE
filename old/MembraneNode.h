//=================================
// include guard
#ifndef MEMBRANE_NODE_H_
#define MEMBRANE_NODE_H_

//=================================
// forward declared dependencies

//=================================
// included dependencies
#include "NodeType.h"

//=================================
// class definition
class MembraneNode : public NodeType {

    // size() = node.membrane_count
    thrust::device_vector<bool> isFixed;

    double mass{ nodeData.defaultMass };
}


#endif