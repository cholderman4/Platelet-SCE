#include "SystemStructures.h"
#include "PlatletSystem.h"
#include "Spring_Force.h"
#include "functor_spring_force.h"

// ******************************************************
// This function is node-based. That is, for each node, it
// loops through all the edges connected to it.
void Spring_Force(
    MembraneNode& memNode, 
    SpringEdge& springEdge, 
    GeneralParams& generalParams) {

    thrust::counting_iterator<unsigned> startEdgeIter(0);

    thrust::for_each(
        startEdgeIter,
        startEdgeIter + generalParams.memNodeCount,
        functor_spring_force(
            thrust::raw_pointer_cast(memNode.pos_x.data()),
            thrust::raw_pointer_cast(memNode.pos_y.data()),
            thrust::raw_pointer_cast(memNode.pos_z.data()),

            thrust::raw_pointer_cast(memNode.force_x.data()),
            thrust::raw_pointer_cast(memNode.force_y.data()),
            thrust::raw_pointer_cast(memNode.force_z.data()),

            thrust::raw_pointer_cast(springEdge.nodeID_L.data()),
            thrust::raw_pointer_cast(springEdge.nodeID_R.data()),
            thrust::raw_pointer_cast(springEdge.len_0.data()),
            thrust::raw_pointer_cast(memNode.springConnections.data()),
            thrust::raw_pointer_cast(memNode.numConnectedSprings.data()),
            
            generalParams.maxConnectedSpringCount,
            generalParams.memSpringStiffness) );
    
}
// ******************************************************
// This function is spring-based. For each spring, it assigns forces 
