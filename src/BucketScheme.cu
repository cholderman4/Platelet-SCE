
#include "BucketScheme.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "function_extend.h"
#include "functor_bucket_indexer.h"
#include "functor_neighbor.h"
#include "NodeData.h"

BucketScheme::BucketScheme(NodeData& _nodeData) :
    nodeData(_nodeData) {};


unsigned* BucketScheme::getDevPtrGlobalNode_ID_expanded() {
    return thrust::raw_pointer_cast(globalNode_ID_expanded.data());
}


unsigned* BucketScheme::getDevPtrKeyBegin() {
    return thrust::raw_pointer_cast(keyBegin.data());
}


unsigned* BucketScheme::getDevPtrKeyEnd() {
    return thrust::raw_pointer_cast(keyEnd.data());
}


thrust::device_vector<unsigned>::iterator BucketScheme::getIteratorBucketID() {
    return bucket_ID.begin();
}


void BucketScheme::execute() {

    initialize_bucket_dimensions();

    set_bucket_grids();

    assign_nodes_to_buckets();

    extend_to_bucket_neighbors();
}

void BucketScheme::initialize_bucket_dimensions() {
    // Should be able to only do membrane nodes.
    min_x = ( *(thrust::min_element(nodeData.pos_x.begin(), nodeData.pos_x.end())));
    min_y = ( *(thrust::min_element(nodeData.pos_y.begin(), nodeData.pos_y.end())));
    min_z = ( *(thrust::min_element(nodeData.pos_z.begin(), nodeData.pos_z.end())));

    max_x = ( *(thrust::max_element(nodeData.pos_x.begin(), nodeData.pos_x.end())));
    max_y = ( *(thrust::max_element(nodeData.pos_y.begin(), nodeData.pos_y.end())));
    max_z = ( *(thrust::max_element(nodeData.pos_z.begin(), nodeData.pos_z.end())));

    double buffer{ 0.0 };
    min_x = min_x - buffer;
    min_y = min_y - buffer;
    min_z = min_z - buffer;

    max_x = max_x + buffer;
    max_y = max_y + buffer;
    max_z = max_z + buffer;    
}

void BucketScheme::set_bucket_grids() {
    bucketCount_x = ceil( (max_x - min_x) / gridSpacing ) + 1;
    bucketCount_y = ceil( (max_y - min_y) / gridSpacing ) + 1;
    bucketCount_z = ceil( (max_z - min_z) / gridSpacing ) + 1;
    
    unsigned newBucketCount_total = bucketCount_x * bucketCount_y * bucketCount_z;

    if ( newBucketCount_total != bucketCount_total ) {

        //double amount of buckets in case of resizing networks
        bucketCount_total = newBucketCount_total;
    
        keyBegin.resize(bucketCount_total);
        keyEnd.resize(bucketCount_total);
    }

    thrust::fill(keyBegin.begin(), keyBegin.end(), 0);
    thrust::fill(keyEnd.begin(), keyEnd.end(), 0);
}

void BucketScheme::assign_nodes_to_buckets() {
    thrust::counting_iterator<unsigned> start(0);
    // takes counting iterator and coordinates
    // return tuple of keys and values
    // transform the points to their bucket indices



    thrust::transform(
        start, 
        start + nodeData.getNodeCount(),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                bucket_ID.begin(),
                globalNode_ID.begin())),
        functor_bucket_indexer(
            min_x, min_y, min_z,
            max_x, max_y, max_z,
            
            bucketCount_x,
            bucketCount_y,
            bucketCount_z,
            gridSpacing,

            thrust::raw_pointer_cast(nodeData.pos_x.data()),
            thrust::raw_pointer_cast(nodeData.pos_y.data()),
            thrust::raw_pointer_cast(nodeData.pos_z.data())));

    // test sorting by node instead of bucket index
    // not sure if necessary
    thrust::sort_by_key(
        globalNode_ID.begin(),
        globalNode_ID.begin() + nodeData.getNodeCount(),
        bucket_ID.begin());

    endIndexBucketKeys = nodeData.getNodeCount();
}

void BucketScheme::extend_to_bucket_neighbors() {
    //memory is already allocated.
	unsigned endIndexExpanded = (endIndexBucketKeys) * 27;
	

	//test for removing copies.
	unsigned valuesCount = globalNode_ID.size();
	thrust::fill(bucket_ID_expanded.begin(), bucket_ID_expanded.end(), 0);
	thrust::fill(globalNode_ID_expanded.begin(), globalNode_ID_expanded.end(), 0);


	
	// beginning of constant iterator
	
	thrust::constant_iterator<unsigned> first(27);
	
	// end of constant iterator.
	// the plus sign only indicate movement of position, not value.
	// e.g. movement is 5 and first iterator is initialized as 9
	// result array is [9,9,9,9,9];
    
    // this is NOT numerical addition!
	thrust::constant_iterator<unsigned> last = first + (endIndexBucketKeys); 

	expand(first, last,
		thrust::make_zip_iterator(
			thrust::make_tuple(
				bucket_ID.begin(),
				globalNode_ID.begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				bucket_ID_expanded.begin(),
                globalNode_ID_expanded.begin())));

	thrust::counting_iterator<unsigned> countingBegin(0);
 
	thrust::transform(
        // Input begin.
		thrust::make_zip_iterator(
			thrust::make_tuple(
				bucket_ID_expanded.begin(),
                countingBegin)),
        
        // Input end.
		thrust::make_zip_iterator(
			thrust::make_tuple(
				bucket_ID_expanded.begin(),
				countingBegin)) + endIndexExpanded,
        
        // Output begin.
        bucket_ID_expanded.begin(),

        // Functor
		functor_neighbor(
			bucketCount_x,
			bucketCount_y,
            bucketCount_z)); 
            
    
	thrust::stable_sort_by_key(
		bucket_ID_expanded.begin(),
		bucket_ID_expanded.end(),
		globalNode_ID_expanded.begin());


	thrust::counting_iterator<unsigned> search_begin(0);

	thrust::lower_bound(
		bucket_ID_expanded.begin(),
		bucket_ID_expanded.end(), 
		search_begin,
		search_begin + bucketCount_total,
		keyBegin.begin());

	thrust::upper_bound(
		bucket_ID_expanded.begin(),
		bucket_ID_expanded.end(),
		search_begin,
		search_begin + bucketCount_total,
        keyEnd.begin());
}