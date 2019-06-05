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
        thrust::make_zip_iterator(
            thrust::make_tuple(
                startEdgeIter,
                node.isFixed.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                startEdgeIter,
                node.isFixed.begin())) + node.membrane_count,
        functor_spring_force(
            thrust::raw_pointer_cast(node.pos_x.data()),
            thrust::raw_pointer_cast(node.pos_y.data()),
            thrust::raw_pointer_cast(node.pos_z.data()),

            thrust::raw_pointer_cast(node.force_x.data()),
            thrust::raw_pointer_cast(node.force_y.data()),
            thrust::raw_pointer_cast(node.force_z.data()),

            thrust::raw_pointer_cast(node.isFixed.data()),

            thrust::raw_pointer_cast(node.connectedSpringID.data()),
            thrust::raw_pointer_cast(node.connectedSpringCount.data()),

            node.maxConnectedSpringCount,

            thrust::raw_pointer_cast(springEdge.nodeID_L.data()),
            thrust::raw_pointer_cast(springEdge.nodeID_R.data()),
            thrust::raw_pointer_cast(springEdge.len_0.data()),
            
            springEdge.stiffness) );    
}