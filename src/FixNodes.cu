#include "FixNodes.h"

#include <thrust/for_each.h>

#include "functor_fix_nodes.h"
#include "NodeData.h"


FixNodes::FixNodes(NodeData& _nodeData) :
    NodeOperation(_nodeData) {};


void FixNodes::execute() {
    
    thrust::for_each(
        fixedNodes.begin(), 
        fixedNodes.end(),
        functor_fix_nodes(
            thrust::raw_pointer_cast(nodeData.force_x.data()),
            thrust::raw_pointer_cast(nodeData.force_y.data()),
            thrust::raw_pointer_cast(nodeData.force_z.data())));

}