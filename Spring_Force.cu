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
        thrust::make_zip_iterator(
            thrust::make_tuple(
                startEdgeIter,
                memNode.isFixed.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                startEdgeIter,
                memNode.isFixed.begin())) + memNode.count,
        functor_spring_force(
            thrust::raw_pointer_cast(memNode.pos_x.data()),
            thrust::raw_pointer_cast(memNode.pos_y.data()),
            thrust::raw_pointer_cast(memNode.pos_z.data()),

            thrust::raw_pointer_cast(memNode.force_x.data()),
            thrust::raw_pointer_cast(memNode.force_y.data()),
            thrust::raw_pointer_cast(memNode.force_z.data()),

            thrust::raw_pointer_cast(memNode.isFixed.data()),

            thrust::raw_pointer_cast(memNode.connectedSpringID.data()),
            thrust::raw_pointer_cast(memNode.connectedSpringCount.data()),

            memNode.maxConnectedSpringCount,

            thrust::raw_pointer_cast(springEdge.nodeID_L.data()),
            thrust::raw_pointer_cast(springEdge.nodeID_R.data()),
            thrust::raw_pointer_cast(springEdge.len_0.data()),
            
            springEdge.stiffness) );
    
}
// ******************************************************
// This function is spring-based. For each spring, it assigns forces 
