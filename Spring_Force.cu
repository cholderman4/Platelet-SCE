#include "SystemStructures.h"
#include "PlatletSystem.h"
#include "Spring_Force.h"
#include "functor_spring_force.h"

// ******************************************************
// This function is node-based. That is, for each node, it
// loops through all the edges connected to it.
void Spring_Force(
    Node& node, 
    SpringEdge& springEdge, 
    GeneralParams& generalParams) {

    thrust::counting_iterator<unsigned> startEdgeIter(0);

    thrust::for_each(
        startEdgeIter,
        startEdgeIter + generalParams.memNodeCount,
        functor_spring_force(
            thrust::raw_pointer_cast(node.pos_x.data()),
            thrust::raw_pointer_cast(node.pos_y.data()),
            thrust::raw_pointer_cast(node.pos_z.data()),

            thrust::raw_pointer_cast(node.force_x.data()),
            thrust::raw_pointer_cast(node.force_y.data()),
            thrust::raw_pointer_cast(node.force_z.data()),

            thrust::raw_pointer_cast(springEdge.nodeID_L.data()),
            thrust::raw_pointer_cast(springEdge.nodeID_R.data()),
            thrust::raw_pointer_cast(springEdge.len_0.data()),
            thrust::raw_pointer_cast(springEdge.nodeConnections.data()),
            thrust::raw_pointer_cast(springEdge.nodeDegree.data()),
            
            generalParams.maxConnectedSpringCount) );
    
}
// ******************************************************
// This function is spring-based. For each spring, it assigns forces 
