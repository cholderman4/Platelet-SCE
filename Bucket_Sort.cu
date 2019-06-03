#include "Bucket_Sort.h"
#include "PlatletSystem.h"

#include "functor_neighbor.h"
#include "functor_bucket_indexer.h"
#include "function_extend.h"


void initialize_bucket_dimensions(
    MembraneNode& memNode,
    Node& intNode,
    DomainParams& domainParams) {

    // Should be able to only do membrane nodes.
    double minMembrane_x = ( *(thrust::min_element(memNode.pos_x.begin(), memNode.pos_x.end())));
    double minMembrane_y = ( *(thrust::min_element(memNode.pos_y.begin(), memNode.pos_y.end())));
    double minMembrane_z = ( *(thrust::min_element(memNode.pos_z.begin(), memNode.pos_z.end())));

    double maxMembrane_x = ( *(thrust::max_element(memNode.pos_x.begin(), memNode.pos_x.end())));
    double maxMembrane_y = ( *(thrust::max_element(memNode.pos_y.begin(), memNode.pos_y.end())));
    double maxMembrane_z = ( *(thrust::max_element(memNode.pos_z.begin(), memNode.pos_z.end())));

    double minInternal_x = ( *(thrust::min_element(intNode.pos_x.begin(), intNode.pos_x.end())));
    double minInternal_y = ( *(thrust::min_element(intNode.pos_y.begin(), intNode.pos_y.end())));
    double minInternal_z = ( *(thrust::min_element(intNode.pos_z.begin(), intNode.pos_z.end())));

    double maxInternal_x = ( *(thrust::max_element(intNode.pos_x.begin(), intNode.pos_x.end())));
    double maxInternal_y = ( *(thrust::max_element(intNode.pos_y.begin(), intNode.pos_y.end())));
    double maxInternal_z = ( *(thrust::max_element(intNode.pos_z.begin(), intNode.pos_z.end())));

    double buffer{ 0.0 };
    domainParams.min_x = min( minMembrane_x, minInternal_x ) - buffer;
    domainParams.min_y = min( minMembrane_y, minInternal_y ) - buffer;
    domainParams.min_z = min( minMembrane_z, minInternal_z ) - buffer;

    domainParams.max_x = max( minMembrane_x, minInternal_x ) + buffer;
    domainParams.max_y = max( minMembrane_y, minInternal_y ) + buffer;
    domainParams.max_z = max( minMembrane_z, minInternal_z ) + buffer;   
}


void set_bucket_grids(
    MembraneNode& memNode,
    Node& intNode,
    DomainParams& domainParams,
    BucketScheme& bucketScheme) {

    domainParams.bucketCount_x = ceil( (domainParams.max_x - domainParams.min_x) / domainParams.gridSpacing ) + 1;
    domainParams.bucketCount_y = ceil( (domainParams.max_y - domainParams.min_y) / domainParams.gridSpacing ) + 1;
    domainParams.bucketCount_z = ceil( (domainParams.max_z - domainParams.min_z) / domainParams.gridSpacing ) + 1;

    unsigned newBucketCount_total = domainParams.bucketCount_x * domainParams.bucketCount_y * domainParams.bucketCount_z;

    if ( newBucketCount_total != domainParams.bucketCount_total ) {

        std::cout<<"resetting grid " << std::endl;
        std::cout<<"x-bucket: "<< domainParams.bucketCount_x <<std::endl;
        std::cout<<"y-bucket: "<< domainParams.bucketCount_y <<std::endl;
        std::cout<<"z-bucket: "<< domainParams.bucketCount_z <<std::endl;

        //double amount of buckets in case of resizing networks
        domainParams.bucketCount_total = newBucketCount_total;
        std::cout<<"grid: "<< domainParams.gridSpacing << std::endl;
        std::cout<<"total bucket count: " << domainParams.bucketCount_total <<std::endl;

        bucketScheme.keyBegin.resize(domainParams.bucketCount_total);
        bucketScheme.keyEnd.resize(domainParams.bucketCount_total);
    }
    thrust::fill(bucketScheme.keyBegin.begin(), bucketScheme.keyBegin.end(), 0);
    thrust::fill(bucketScheme.keyEnd.begin(), bucketScheme.keyEnd.end(), 0);
}



void assign_nodes_to_buckets(
    MembraneNode& memNode,
    Node& intNode,
    DomainParams& domainParams,
    BucketScheme& bucketScheme) {

    thrust::counting_iterator<unsigned> indexNodeBegin(0);
    // takes counting iterator and coordinates
    // return tuple of keys and values
    // transform the points to their bucket indices



    thrust::transform(
        indexNodeBegin, 
        indexNodeBegin + memNode.count + intNode.count,
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

            thrust::raw_pointer_cast(memNode.pos_x.data()),
            thrust::raw_pointer_cast(memNode.pos_y.data()),
            thrust::raw_pointer_cast(memNode.pos_z.data()),
            memNode.count,

            thrust::raw_pointer_cast(intNode.pos_x.data()),
            thrust::raw_pointer_cast(intNode.pos_y.data()),
            thrust::raw_pointer_cast(intNode.pos_z.data())));

    // test sorting by node instead of bucket index
    // not sure if necessary
    thrust::sort_by_key(
        bucketScheme.globalNode_ID.begin(),
        bucketScheme.globalNode_ID.begin() + memNode.count + intNode.count,
        bucketScheme.bucket_ID.begin());

    bucketScheme.endIndexBucketKeys = memNode.count + intNode.count;
}


void extend_to_bucket_neighbors(
    MembraneNode& memNode,
    Node& intNode,
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

std::cerr << "expand function starting\n";
	expand(first, last,
		thrust::make_zip_iterator(
			thrust::make_tuple(
				bucketScheme.bucket_ID.begin(),
				bucketScheme.globalNode_ID.begin())),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				bucketScheme.bucket_ID_expanded.begin(),
                bucketScheme.globalNode_ID_expanded.begin())));
std::cerr << "expand function finished\n";

	thrust::counting_iterator<unsigned> countingBegin(0);
 
std::cerr << "transform functor_neighbor\n";
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
            
    
std::cerr << "sort by key\n";
	thrust::stable_sort_by_key(
		bucketScheme.bucket_ID_expanded.begin(),
		bucketScheme.bucket_ID_expanded.end(),
		bucketScheme.globalNode_ID_expanded.begin());


	thrust::counting_iterator<unsigned> search_begin(0);

std::cerr << "Initializing key begin/end\n";
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
std::cerr << "expand function finished\n";        
}