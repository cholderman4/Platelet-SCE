#include "AdvancePositionsByForce.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "functor_advance_pos.h"
#include "NodeData.h"
#include "NodeOperation.h"
#include "RandomUtil.h"


AdvancePositionsByForce::AdvancePositionsByForce(NodeData& _nodeData) :
	NodeOperation(_nodeData) {};

void AdvancePositionsByForce::execute() {
    /****************************************************************************************** 
		Random number generation.  */
		unsigned _seed = rand();
    	thrust::device_vector<double> gaussianData;
    	gaussianData.resize(indexEnd - indexBegin); 
		thrust::counting_iterator<unsigned> index_sequence_begin(_seed);

		thrust::transform(
			thrust::device, 
			index_sequence_begin, 
			index_sequence_begin + (indexEnd - indexBegin),
			gaussianData.begin(), 
			psrunifgen(-1.0, 1.0));
		
	/****************************************************************************************** */

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

			/****************************************************************************************** 
			Random number generation.  */
			unsigned _seed = rand();
			thrust::device_vector<double> gaussianData;
			gaussianData.resize(end - begin); 
			thrust::counting_iterator<unsigned> index_sequence_begin(_seed);

			thrust::transform(
				thrust::device, 
				index_sequence_begin, 
				index_sequence_begin + (end - begin),
				gaussianData.begin(), 
				psrunifgen(-1.0, 1.0));
			/****************************************************************************************** */

	
			thrust::transform(
				// Input vector #1
				thrust::make_zip_iterator(
					thrust::make_tuple(
						iteratorStart,
						nodeData.pos_x.begin(),
						nodeData.pos_y.begin(),
						nodeData.pos_z.begin())) + begin,
	
				thrust::make_zip_iterator(
					thrust::make_tuple(
						iteratorStart,
						nodeData.pos_x.begin(),
						nodeData.pos_y.begin(),
						nodeData.pos_z.begin())) + end,
		
				// Input vector #2
				thrust::make_zip_iterator(
					thrust::make_tuple(
						gaussianData.begin(),
						nodeData.force_x.begin() + begin,
						nodeData.force_y.begin() + begin,
						nodeData.force_z.begin() + begin)),
	
				// Output vector
				thrust::make_zip_iterator(
					thrust::make_tuple(
						nodeData.pos_x.begin(),
						nodeData.pos_y.begin(),
						nodeData.pos_z.begin(),
						nodeData.velocity.begin())) + begin,
				// Functor + parameter call
				functor_advance_pos(
					dt,
					viscousDamp,
					temperature,
					kB,
					nodeData.defaultMass));

			nNodesTransformed += (end - begin);
			
			gaussianData.clear();
			gaussianData.shrink_to_fit();
		}
}