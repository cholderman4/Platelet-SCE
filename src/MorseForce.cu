#include "MorseForce.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "BucketScheme.h"
#include "functor_morse_force.h"
#include "NodeData.h"

MorseForce::MorseForce(NodeData& _nodeData, BucketScheme& _bucketScheme) :
    bucketScheme(_bucketScheme),
    NodeTypeOperation(_nodeData) {};



void MorseForce::getParameterKeys(ParameterManager& parameterManager) {
    for(auto i = 0; i < nodeInteractionA.size(); ++i) {
        parameterManager.setValue("morse_U", 0.0, nodeInteractionA[i], nodeInteractionB[i]);
        parameterManager.setValue("morse_P", 0.0, nodeInteractionA[i], nodeInteractionB[i]);
        parameterManager.setValue("morse_R_eq", 0.0, nodeInteractionA[i], nodeInteractionB[i]);
    }
}


void MorseForce::setParameterValues(ParameterManager& parameterManager) {
    for(auto i = 0; i < nodeInteractionA.size(); ++i) {
        parameterManager.findValue("morse_U", U[i], nodeInteractionA[i], nodeInteractionB[i]);
        parameterManager.findValue("morse_P", P[i], nodeInteractionA[i], nodeInteractionB[i]);
        parameterManager.findValue("morse_R_eq", R_eq[i], nodeInteractionA[i], nodeInteractionB[i]);
    }
}



void MorseForce::execute() {

    // Operations that calculate energy must first clear energy values.
    thrust::fill(nodeData.energy.begin(), nodeData.energy.end(), 0.0);
    energy = 0.0;

    unsigned end = nodeData.pos_x.size();

    thrust::counting_iterator<unsigned> iteratorStart(0);

    thrust::transform(
        // Input begin.
        thrust::make_zip_iterator(
            thrust::make_tuple(
                iteratorStart,
                bucketScheme.getIteratorBucketID())),
                // bucketScheme.bucket_ID.begin())),

        // Input end.
        thrust::make_zip_iterator(
            thrust::make_tuple(
                iteratorStart,
                bucketScheme.getIteratorBucketID())) + end,
                // bucketScheme.bucket_ID.begin())) + end,

        // Output
        nodeData.energy.begin(),

        functor_morse_force(
            thrust::raw_pointer_cast(nodeData.pos_x.data()),
            thrust::raw_pointer_cast(nodeData.pos_y.data()),
            thrust::raw_pointer_cast(nodeData.pos_z.data()),

            thrust::raw_pointer_cast(nodeData.force_x.data()),
            thrust::raw_pointer_cast(nodeData.force_y.data()),
            thrust::raw_pointer_cast(nodeData.force_z.data()),

            thrust::raw_pointer_cast(indicesBegin.data()),
            thrust::raw_pointer_cast(indicesEnd.data()),
            nNodeTypes,

            bucketScheme.getDevPtrGlobalNode_ID_expanded(),
            bucketScheme.getDevPtrKeyBegin(),
            bucketScheme.getDevPtrKeyEnd(),

            thrust::raw_pointer_cast(U.data()),
            thrust::raw_pointer_cast(P.data()),
            thrust::raw_pointer_cast(R_eq.data())) );

    energy += thrust::reduce(
        nodeData.energy.begin(),
        nodeData.energy.end());

}