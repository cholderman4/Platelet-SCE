#include "SystemStructures.h"
#include "PlatletSystem.h"
#include "LJ_Force.h"
#include "functor_LJ_force.h"

// ******************************************************

void LJ_Force(
    Node& node, 
    SpringEdge& springEdge, 
    GeneralParams& generalParams) {

    thrust::counting_iterator<unsigned> startEdgeIter(0);

    thrust::for_each(
        startEdgeIter,
        startEdgeIter + generalParams.memNodeCount + generalParams.internalNodeCount,
        functor_spring_force(
            thrust::raw_pointer_cast(node.pos_x.data()),
            thrust::raw_pointer_cast(node.pos_y.data()),
            thrust::raw_pointer_cast(node.pos_z.data()),

            thrust::raw_pointer_cast(node.force_x.data()),
            thrust::raw_pointer_cast(node.force_y.data()),
            thrust::raw_pointer_cast(node.force_z.data()),
        
            generalParams.memNodeCount + generalParams.internalNodeCount) );
    
}
