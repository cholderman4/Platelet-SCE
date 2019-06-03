#include "SystemStructures.h"
#include "PlatletSystem.h"
#include "LJ_Force.h"
#include "functor_LJ_force.h"

// ******************************************************

void LJ_Force(
    MembraneNode& memNode,
    Node& intNode,
    BucketScheme& bucketScheme, 
    GeneralParams& generalParams) {

    thrust::counting_iterator<unsigned> startNodeIter(0);

    thrust::for_each(
        thrust::make_zip_iterator(
            thrust::make_tuple(
                startNodeIter,
                bucketScheme.bucket_ID.begin())),

        thrust::make_zip_iterator(
            thrust::make_tuple(
                startNodeIter,
                bucketScheme.bucket_ID.begin()))+ memNode.count + intNode.count,

        functor_LJ_force(
            thrust::raw_pointer_cast(memNode.pos_x.data()),
            thrust::raw_pointer_cast(memNode.pos_y.data()),
            thrust::raw_pointer_cast(memNode.pos_z.data()),

            thrust::raw_pointer_cast(memNode.force_x.data()),
            thrust::raw_pointer_cast(memNode.force_y.data()),
            thrust::raw_pointer_cast(memNode.force_z.data()),
        
            thrust::raw_pointer_cast(memNode.isFixed.data()),

            memNode.count,

            thrust::raw_pointer_cast(intNode.pos_x.data()),
            thrust::raw_pointer_cast(intNode.pos_y.data()),
            thrust::raw_pointer_cast(intNode.pos_z.data()),

            thrust::raw_pointer_cast(intNode.force_x.data()),
            thrust::raw_pointer_cast(intNode.force_y.data()),
            thrust::raw_pointer_cast(intNode.force_z.data()),
        
            intNode.count,

            thrust::raw_pointer_cast(bucketScheme.globalNode_ID_expanded.data()),
            thrust::raw_pointer_cast(bucketScheme.keyBegin.data()),
            thrust::raw_pointer_cast(bucketScheme.keyEnd.data()),


            generalParams.U_II,
            generalParams.P_II,
            generalParams.R_eq_II,
            generalParams.U_MI,
            generalParams.P_MI,
            generalParams.R_eq_MI) );
    
}
