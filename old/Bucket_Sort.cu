#include "Bucket_Sort.h"
#include "PlatletSystem.h"

#include "functor_neighbor.h"
#include "functor_bucket_indexer.h"
#include "function_extend.h"


void initialize_bucket_dimensions(
    Node& node,
    DomainParams& domainParams) {

    // Should be able to only do membrane nodes.
    double min_x = ( *(thrust::min_element(node.pos_x.begin(), node.pos_x.end())));
    double min_y = ( *(thrust::min_element(node.pos_y.begin(), node.pos_y.end())));
    double min_z = ( *(thrust::min_element(node.pos_z.begin(), node.pos_z.end())));

    double max_x = ( *(thrust::max_element(node.pos_x.begin(), node.pos_x.end())));
    double max_y = ( *(thrust::max_element(node.pos_y.begin(), node.pos_y.end())));
    double max_z = ( *(thrust::max_element(node.pos_z.begin(), node.pos_z.end())));

    double buffer{ 0.0 };
    domainParams.min_x = min_x - buffer;
    domainParams.min_y = min_y - buffer;
    domainParams.min_z = min_z - buffer;

    domainParams.max_x = max_x + buffer;
    domainParams.max_y = max_y + buffer;
    domainParams.max_z = max_z + buffer;   
}


void set_bucket_grids(
    Node& node,
    DomainParams& domainParams,
    BucketScheme& bucketScheme) {

    domainParams.bucketCount_x = ceil( (domainParams.max_x - domainParams.min_x) / domainParams.gridSpacing ) + 1;
    domainParams.bucketCount_y = ceil( (domainParams.max_y - domainParams.min_y) / domainParams.gridSpacing ) + 1;
    domainParams.bucketCount_z = ceil( (domainParams.max_z - domainParams.min_z) / domainParams.gridSpacing ) + 1;
    
    unsigned newBucketCount_total = domainParams.bucketCount_x * domainParams.bucketCount_y * domainParams.bucketCount_z;

    if ( newBucketCount_total != domainParams.bucketCount_total ) {

        //double amount of buckets in case of resizing networks
        domainParams.bucketCount_total = newBucketCount_total;
    
        bucketScheme.keyBegin.resize(domainParams.bucketCount_total);
        bucketScheme.keyEnd.resize(domainParams.bucketCount_total);
    }

    thrust::fill(bucketScheme.keyBegin.begin(), bucketScheme.keyBegin.end(), 0);
    thrust::fill(bucketScheme.keyEnd.begin(), bucketScheme.keyEnd.end(), 0);
}



void assign_nodes_to_buckets(
    Node& node,
    DomainParams& domainParams,
    BucketScheme& bucketScheme) {

    thrust::counting_iterator<unsigned> indexNodeBegin(0);
    // takes counting iterator and coordinates
    // return tuple of keys and values
    // transform the points to their bucket indices



    thrust::transform(
        indexNodeBegin, 
        indexNodeBegin + node.total_count,
        thrust::make_zip_iterator(
            thrust::make_tuple(
                bucketScheme.bucket_ID.begin(),
                bucketScheme.globalNode_ID.begin())),
        functor_bucket_indexer(
            domainParams.min_x, domainParams.min_y, domainParams.min_z,
            domainParams.max_x, domainParams.max_y, domainParams.max_z,
            
            domainParams.bucketCount_x,
            domainParams.bucketCount_y,
            domainParams.bucketCount_z,
            domainParams.gridSpacing,

            thrust::raw_pointer_cast(node.pos_x.data()),
            thrust::raw_pointer_cast(node.pos_y.data()),
            thrust::raw_pointer_cast(node.pos_z.data())));

    // test sorting by node instead of bucket index
    // not sure if necessary
    thrust::sort_by_key(
        bucketScheme.globalNode_ID.begin(),
        bucketScheme.globalNode_ID.begin() + node.total_count,
        bucketScheme.bucket_ID.begin());

    bucketScheme.endIndexBucketKeys = node.total_count;
}


void extend_to_bucket_neighbors(
    Node& node,
    DomainParams& domainParams,
    BucketScheme& bucketScheme) {

    //memory is already allocated.
	unsigned endIndexExpanded = (bucketScheme.endIndexBucketKeys) * 27;
	

	//test for removing copies.
	unsigned valuesCount = bucketScheme.globalNode_ID.size();
	thrust::fill(bucketScheme.bucket_ID_expanded.begin(),bucketScheme.bucket_ID_expanded.end(),0);
	thrust::fill(bucketScheme.globalNode_ID_expanded.begin(),bucketScheme.globalNode_ID_expanded.end(),0);


	
	// beginning of constant iterator
	
	thrust::constant_iterator<unsigned> first(27);
	
	// end of constant iterator.
	// the plus sign only indicate movement of position, not value.
	// e.g. movement is 5 and first iterator is initialized as 9
	// result array is [9,9,9,9,9];
    
    // this is NOT numerical addition!
	thrust::constant_iterator<unsigned> last = first + (bucketScheme.endIndexBucketKeys); 

	expand(first, last,
		thrust::make_zip_iterator(
			thrust::make_tuple(
				bucketScheme.bucket_ID.begin(),
				bucketScheme.globalNode_ID.begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				bucketScheme.bucket_ID_expanded.begin(),
                bucketScheme.globalNode_ID_expanded.begin())));

	thrust::counting_iterator<unsigned> countingBegin(0);
 
	thrust::transform(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				bucketScheme.bucket_ID_expanded.begin(),
				countingBegin)),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				bucketScheme.bucket_ID_expanded.begin(),
				countingBegin)) + endIndexExpanded,
		
		bucketScheme.bucket_ID_expanded.begin(),
		functor_neighbor(
			domainParams.bucketCount_x,
			domainParams.bucketCount_y,
            domainParams.bucketCount_z)); 
            
    
	thrust::stable_sort_by_key(
		bucketScheme.bucket_ID_expanded.begin(),
		bucketScheme.bucket_ID_expanded.end(),
		bucketScheme.globalNode_ID_expanded.begin());


	thrust::counting_iterator<unsigned> search_begin(0);

	thrust::lower_bound(
		bucketScheme.bucket_ID_expanded.begin(),
		bucketScheme.bucket_ID_expanded.end(), 
		search_begin,
		search_begin + domainParams.bucketCount_total,
		bucketScheme.keyBegin.begin());

	thrust::upper_bound(
		bucketScheme.bucket_ID_expanded.begin(),
		bucketScheme.bucket_ID_expanded.end(),
		search_begin,
		search_begin + domainParams.bucketCount_total,
        bucketScheme.keyEnd.begin());
}