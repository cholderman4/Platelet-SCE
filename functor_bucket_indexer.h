#ifndef FUNCTOR_BUCKET_INDEXER_H_
#define FUNCTOR_BUCKET_INDEXER_H_

#include "SystemStructures.h"

struct functor_bucket_indexer {
	double min_x;
	double min_y;
	double min_z;
	double max_x;
	double max_y;
	double max_z;

	unsigned bucketCount_x;
	unsigned bucketCount_y;
	unsigned bucketCount_z;
	double gridSpacing;

    double* vec_pos_x;
    double* vec_pos_y;
    double* vec_pos_z;

	__host__ __device__

	functor_bucket_indexer(
		double& _min_x,
		double& _min_y,
		double& _min_z,
		double& _max_x,
		double& _max_y,
		double& _max_z,

		unsigned _bucketCount_x,
		unsigned _bucketCount_y,
		unsigned _bucketCount_z,
		double _gridSpacing,
        
        double* _vec_pos_x,
        double* _vec_pos_y,
        double* _vec_pos_z) :

		min_x(_min_x),
		min_y(_min_y),
		min_z(_min_z),
		max_x(_max_x),
		max_y(_max_y),
		max_z(_max_z),

		bucketCount_x(_bucketCount_x),
		bucketCount_y(_bucketCount_y),
		bucketCount_z(_bucketCount_z),
		gridSpacing(_gridSpacing),
        
        vec_pos_x(_vec_pos_x),
        vec_pos_y(_vec_pos_y),
        vec_pos_z(_vec_pos_z) {}

	__device__ 
	Tuu operator()(const unsigned& id) {

        double pos_x = vec_pos_x[id];
        double pos_y = vec_pos_y[id];
        double pos_z = vec_pos_z[id];


        unsigned x = static_cast<unsigned>((pos_x - min_x) / gridSpacing);
        unsigned y = static_cast<unsigned>((pos_y - min_y) / gridSpacing);
        unsigned z = static_cast<unsigned>((pos_z - min_z) / gridSpacing);


        // return the bucket's linear index and node's global index
        //return thrust::make_tuple(z * XSize * YSize + y * XSize + x, thrust::get<4>(v));
        unsigned bucket =   z * bucketCount_x * bucketCount_y 
                            + y * bucketCount_x 
                            + x;
        //try to make it so bucket does not return unsigned32Max
        /* if (bucket == ULONG_MAX) {
            bucket = 0;
        } */
        return thrust::make_tuple(bucket, id);

	}
};

#endif