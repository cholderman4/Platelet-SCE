#include "SpringForce.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "functor_spring_force.h"
#include "NodeData.h"
#include "NodeTypeOperation.h"
#include "ParameterManager.h"


SpringForce::SpringForce(NodeData& _nodeData) :
    NodeTypeOperation(_nodeData) {};



void SpringForce::getParameterKeys(ParameterManager& parameterManager) {
    parameterManager.setValue("springStiffness", stiffness);
    parameterManager.setValue("nMaxSpringsConnectedToNode", nMaxSpringsConnectedToNode);
}


void SpringForce::setParameterValues(ParameterManager& parameterManager) {
    stiffness = parameterManager.getValue("springStiffness");

    // Should include some checking somewhere.
    nMaxSpringsConnectedToNode =  static_cast<unsigned>(parameterManager.getValue("nMaxSpringsConnectedToNode"));
}



void SpringForce::execute() {

    // Operations that calculate energy must first clear energy values.
    thrust::fill(nodeData.energy.begin(), nodeData.energy.end(), 0.0);
    energy = 0.0;

    thrust::counting_iterator<unsigned> iteratorStart(0);
    
    unsigned nNodesTransformed = 0;

    for (int i = 0; i < nNodeTypes; ++i) {
        // Each for loop iteration corresponds to a new continuous chunk of data.
        unsigned begin = indicesBegin[i];

        // Find the end of the chunk.
        bool isConnected = true;
        while ( isConnected ) {
            if (i+1 < nNodeTypes) {
                if( indicesEnd[i] == indicesBegin[i+1] ) {
                    ++i;
                } else {
                    isConnected = false;
                } 
            } else {
                isConnected = false;
            }
        }
        unsigned end = indicesEnd[i];

        thrust::transform(
            // Input begin.
            iteratorStart + begin,
    
            // Input end.
            iteratorStart + end,
    
            // Output.
            nodeData.energy.begin() + begin,
            
            // Functor
            functor_spring_force(
                thrust::raw_pointer_cast(nodeData.pos_x.data()),
                thrust::raw_pointer_cast(nodeData.pos_y.data()),
                thrust::raw_pointer_cast(nodeData.pos_z.data()),
    
                thrust::raw_pointer_cast(nodeData.force_x.data()),
                thrust::raw_pointer_cast(nodeData.force_y.data()),
                thrust::raw_pointer_cast(nodeData.force_z.data()),       
    
                thrust::raw_pointer_cast(springConnectionsByNode.data()),
                thrust::raw_pointer_cast(nSpringsConnectedToNode.data()),
                nMaxSpringsConnectedToNode,
                nNodesTransformed,

                thrust::raw_pointer_cast(nodeConnectionsBySpring.data()),
                thrust::raw_pointer_cast(len_0.data()),

                stiffness) );

        energy += thrust::reduce(
            nodeData.energy.begin() + begin,
            nodeData.energy.begin() + end );

        nNodesTransformed += (end - begin);
    }
}