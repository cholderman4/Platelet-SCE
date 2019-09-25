#ifndef FUNCTOR_SPRING_FORCE_H_
#define FUNCTOR_SPRING_FORCE_H_

#include <cmath>
#include <thrust/tuple.h>


/* Used to advance position based on given velocity vector.
    
/* Input: 	U_CVec3 id_pos: id of node
	                        position coordinates (x, y, z)

			
/* Output:	CVec3:			Updated position coordinates (x, y, z)*/

/* Update:	None	 */
typedef thrust::tuple<unsigned, double, double, double> U_CVec3;
typedef thrust::tuple<double, double, double> CVec3;


struct functor_advance_by_vel : public thrust::unary_function<U_CVec3, CVec3> {
    double dt;
    double vel_x;
    double vel_y;
    double vel_z;    

    __host__ __device__
        functor_advance_by_vel(
            double& _dt,
            double& _vel_x,
            double& _vel_y,
            double& _vel_z) :

        dt(_dt),
        vel_x(_vel_x),
        vel_y(_vel_y),
        vel_z(_vel_z) {}


    __device__
    CVec3 operator()(const U_CVec3& id_pos) {
        // ID of the node being acted on.
        // unsigned idA = thrust::get<0>(u1b1);
        // bool isFixed = thrust::get<1>(u1b1);

        unsigned id = thrust::get<0>(id_pos);
        double pos_x = thrust::get<1>(id_pos);
        double pos_y = thrust::get<2>(id_pos);
        double pos_z = thrust::get<3>(id_pos);
                   
        double newPos_x = pos_x + dt * vel_x;
        double newPos_y = pos_y + dt * vel_y;
        double newPos_z = pos_z + dt * vel_z;

        return thrust::make_tuple(
            newPos_x, 
            newPos_y, 
            newPos_z);      

    } // End operator()
}; // End struct

#endif