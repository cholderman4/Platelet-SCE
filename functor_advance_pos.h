#ifndef FUNCTOR_ADVANCE_POS_H_
#define FUNCTOR_ADVANCE_POS_H_

#include "SystemStructures.h"

/* Used to advance position and velocity from known force
Velocity Verlet Like algorithm used : https://arxiv.org/pdf/1212.1244.pdf */

/* Input: 	CVec3 p3:		position coordinates (x, y, z)

			CVec4 g1f3:		random (gaussian) data from (-1, 1)
							force vector (x, y, z) */

/* Output:	CVec4:			Updated position coordinates (x, y, z)
							velocity magnitude	 */

/* Update:	None	 */


struct functor_advance_pos : public thrust::binary_function<UCVec3, CVec4, CVec4> {
	double dt;
	double viscousDamp;
	double temperature;
	double kB;
	double mass;

	__host__ __device__
		
		functor_advance_pos(
			double& _dt,
			double& _viscousDamp,
			double& _temperature,
			double& _kB,
			double& _mass :

		dt(_dt),
		viscousDamp(_viscousDamp),
		temperature(_temperature),
		kB(_kB),
		mass(_mass) {}

	__device__
		CVec4 operator()(const UCVec3& p3, const CVec4& g1f3) {

			unsigned id = thrust::get<0>(p3);		

			double pos_x = thrust::get<1>(p3);
			double pos_y = thrust::get<2>(p3);
			double pos_z = thrust::get<3>(p3);

			//random data
			double gaussianData = thrust::get<0>(g1f3);

			/* Normally you would have 

					x_(n+1) - x(n) / dt = F / eta + F_b / eta, 
			and 
					F_b = sqrt(2*kb*T*eta/dt) 

			after multiplication, additive noise becomes 
					sqrt(2*kb*t*dt/eta) * N(0,1) 
			*/
		
			double noise = sqrt(2.0 * kB* temperature * dt  / viscousDamp) * gaussianData;

			double acc_x = (thrust::get<1>(g1f3));
			double acc_y = (thrust::get<2>(g1f3));
			double acc_z = (thrust::get<3>(g1f3));


			//update positions
			double newPos_x = locX + (dt/viscousDamp) * (acc_x) + noise;
			double newPos_y = locY + (dt/viscousDamp) * (acc_y) + noise;
			double newPos_z = locZ + (dt/viscousDamp) * (acc_z) + noise;

			double velocity = sqrt((newPos_x - locX) * (newPos_x - locX) + 
									(newPos_y - locY) * (newPos_y - locY) + 
									(newPos_z - locZ) * (newPos_z - locZ));



			return thrust::make_tuple(
									newPos_x, 
									newPos_y, 
									newPos_z, 
									velocity);
	}

};

#endif