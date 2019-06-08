#include "SystemStructures.h"
#include "PlatletSystem.h"
#include "LJ_Force.h"
#include "functor_LJ_force.h"

// ******************************************************

void LJ_Force(
    Node& node,
    BucketScheme& bucketScheme, 
    GeneralParams& generalParams) {

    thrust::counting_iterator<unsigned> startNodeIter(0);

    thrust::transform(
        // Input begin.
        thrust::make_zip_iterator(
            thrust::make_tuple(
                startNodeIter,
                bucketScheme.bucket_ID.begin())),

        // Input end.
        thrust::make_zip_iterator(
            thrust::make_tuple(
                startNodeIter,
                bucketScheme.bucket_ID.begin()))+ node.total_count,

        // Output
        node.energy.begin(),

        functor_LJ_force(
            thrust::raw_pointer_cast(node.pos_x.data()),
            thrust::raw_pointer_cast(node.pos_y.data()),
            thrust::raw_pointer_cast(node.pos_z.data()),

            thrust::raw_pointer_cast(node.force_x.data()),
            thrust::raw_pointer_cast(node.force_y.data()),
            thrust::raw_pointer_cast(node.force_z.data()),
        
            thrust::raw_pointer_cast(node.isFixed.data()),

            node.membrane_count,
            node.interior_count,

            thrust::raw_pointer_cast(bucketScheme.globalNode_ID_expanded.data()),
            thrust::raw_pointer_cast(bucketScheme.keyBegin.data()),
            thrust::raw_pointer_cast(bucketScheme.keyEnd.data()),

            generalParams.U_II,
            generalParams.P_II,
            generalParams.R_eq_II,
            generalParams.U_MI,
            generalParams.P_MI,
            generalParams.R_eq_MI) );

            generalParams.totalEnergy += 
            thrust::reduce(
                node.energy.begin(),
                node.energy.begin() + node.total_count);        
}
