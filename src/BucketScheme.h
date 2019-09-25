#ifndef BUCKET_SCHEME_H_
#define BUCKET_SCHEME_H_


#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "IFunction.h"

class NodeData;

class BucketScheme : public IFunction {

    private:
    NodeData& nodeData;

    // For now, these are scalars, but in future these may be vectors to have one for each platelet.
    double min_x;
    double min_y;
    double min_z;
    double max_x;
    double max_y;
    double max_z;

    double gridSpacing{ 0.200 };

    unsigned bucketCount_x;
    unsigned bucketCount_y;
    unsigned bucketCount_z;
    unsigned bucketCount_total{ 0 };

    thrust::device_vector<unsigned> keyBegin;
    thrust::device_vector<unsigned> keyEnd;

    thrust::device_vector<unsigned> bucket_ID;
    thrust::device_vector<unsigned> globalNode_ID;

    thrust::device_vector<unsigned> bucket_ID_expanded;
    thrust::device_vector<unsigned> globalNode_ID_expanded;

    unsigned endIndexBucketKeys;

    void initialize_bucket_dimensions();

    void set_bucket_grids();

    void assign_nodes_to_buckets();

    void extend_to_bucket_neighbors();

    public:

        BucketScheme(NodeData& _nodeData);

        void execute();

        unsigned* getDevPtrGlobalNode_ID_expanded();
        unsigned* getDevPtrKeyBegin();
        unsigned* getDevPtrKeyEnd();
        thrust::device_vector<unsigned>::iterator getIteratorBucketID();


};

#endif