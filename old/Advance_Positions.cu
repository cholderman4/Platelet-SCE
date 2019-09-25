#include "functor_advance_pos.h"
#include "PlatletSystem.h"
#include "Advance_Positions.h"



double Advance_Positions(
	Node& node,
	GeneralParams& generalParams) {


		double mass = 1.0;
		/* At this point, the previous node location is the same as the current node,
		we can therefore use previous node locations to update nodeLoc. */

		/* ***************************************************************************************** 
		Random number generation.  */
		unsigned _seed = rand();
    	thrust::device_vector<double> gaussianData;
    	gaussianData.resize(node.total_count); //
		thrust::counting_iterator<unsigned> index_sequence_begin(_seed);

		thrust::transform(
			thrust::device, 
			index_sequence_begin, 
			index_sequence_begin + (node.total_count),
			gaussianData.begin(), 
			psrunifgen(-1.0, 1.0));
		
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
					node.pos_z.begin())) + node.total_count,
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

				mass));
		
		gaussianData.clear();
		gaussianData.shrink_to_fit();

	return generalParams.dt;
		//now that nodeLoc is different, we can calculate change and then set previous location
		//to the current location.

}
