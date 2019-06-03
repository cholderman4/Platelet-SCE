#include "functor_advance_pos.h"
#include "PlatletSystem.h"
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
    	gaussianData.resize(node.count); //
		thrust::counting_iterator<unsigned> index_sequence_begin(_seed);

std::cerr << "filling gaussian noise\n";
		thrust::transform(//thrust::device, 
			index_sequence_begin, 
			index_sequence_begin + (node.count),
			gaussianData.begin(), 
			psrunifgen(-1.0, 1.0));
		
		/* ***************************************************************************************** */

		thrust::counting_iterator<unsigned> nodeIndexBegin(0);

std::cerr << "transform: advance_position\n";
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
					node.pos_z.begin())) + node.count,
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

				node.mass));
				
std::cerr << "gaussianData size: " << gaussianData.size() << '\n';
std::cerr << "clearing gaussian data\n";
		// Clear the random data.
		gaussianData.clear();
std::cerr << "cleared\n";
std::cerr << "gaussianData size: " << gaussianData.size() << '\n';

std::cerr << "shrink_to_fit\n";
		gaussianData.shrink_to_fit();

		// Another trick to destroy vector
		// gaussianData.clear();
		// thrust::device_vector<double>().swap(gaussianData);
std::cerr << "gaussian data cleared\n";

	return generalParams.dt;
		//now that nodeLoc is different, we can calculate change and then set previous location
		//to the current location.

}
