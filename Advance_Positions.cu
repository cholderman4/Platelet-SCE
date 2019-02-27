#include "functor_advance_pos.h"
#include "System.h"
#include "Advance_Positions.h"


double Advance_Positions(
	Node& node,
	GeneralParams& generalParams) {


		/* At this point, the previous node location is the same as the current node,
		we can therefore use previous node locations to update nodeLoc. */

		/* ***************************************************************************************** 
		Random number generation.  */
		unsigned _seed = rand();
    	thrust::device_vector<double> gaussianData;
    	gaussianData.resize(generalParams.memNodeCount); //
		thrust::counting_iterator<unsigned> index_sequence_begin(_seed);

    	thrust::transform(thrust::device, index_sequence_begin, index_sequence_begin + (generalParams.maxNodeCount),
		gaussianData.begin(), psrunifgen(-1.0, 1.0));
		
		/* ***************************************************************************************** */

		thrust::counting_iterator<unsigned> nodeIndexBegin(0);

		thrust::transform(
			// Input vector #1
			thrust::make_zip_iterator(
				thrust::make_tuple(
					nodeIndexBegin,
					node.pos_x.begin(),
					node.pos_y.begin(),
					node.pos_z.begin())),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					nodeIndexBegin,
					node.pos_x.begin(),
					node.pos_y.begin(),
					node.pos_z.begin())) + generalParams.memNodeCount,
			// Input vector #2
			thrust::make_zip_iterator(
				thrust::make_tuple(
					gaussianData.begin(),
					node.force_x.begin(),
					node.force_y.begin(),
					node.force_z.begin())),
			// Output vector
			thrust::make_zip_iterator(
				thrust::make_tuple(
					node.pos_x.begin(),
					node.pos_y.begin(),
					node.pos_z.begin(),
					node.velocity.begin())),
			// Functor + parameter call
			functor_advance_pos(
				generalParams.dt,
				generalParams.viscousDamp,
				generalParams.temperature,
				generalParams.kB,
				generalParams.memNodeMass));

		// Clear the random data.
        gaussianData.clear();
        gaussianData.shrink_to_fit();

	return generalParams.dtTemp;
		//now that nodeLoc is different, we can calculate change and then set previous location
		//to the current location.

}
