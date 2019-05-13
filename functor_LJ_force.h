#ifndef FUNCTOR_LJ_FORCE_H_
#define FUNCTOR_LJ_FORCE_H_

#include "SystemStructures.h"


/* Input:   node ID */  

/* Output:  VOID */

/* Update:  node.force_(x,y,z) on given node based on LJ interactions with ALL nodes. */

// This functor does not account for fixed nodes.




__host__ __device__ double LJForceByCoord(double dist, double coordDist) {
    double k = 20;

    return k*(dist - l_0)*coordDist/dist;
}

template <typename T>
__host__ __device__ T norm(T x, T y, T z) {
        return sqrt(x*x + y*y + z*z);
}


struct functor_spring_force : public thrust::unary_function<unsigned, void> {
    double* posVec_x;
    double* posVec_y;
    double* posVec_z;

    double* forceVec_x;
    double* forceVec_y;
    double* forceVec_z;

    unsigned totalNodeCount;


    __host__ __device__
        functor_spring_force(
            double* _posVec_x,
            double* _posVec_y,
            double* _posVec_z,
            
            double* _forceVec_x,
            double* _forceVec_y,
            double* _forceVec_z,

            unsigned& _totalNodeCount) :

        posVec_x(_posVec_x),
        posVec_y(_posVec_y),
        posVec_z(_posVec_z),
        
        forceVec_x(_forceVec_x),
        forceVec_y(_forceVec_y),
        forceVec_z(_forceVec_z),

        totalNodeCount(_totalNodeCount) {}


    __device__
    void operator()(const unsigned idA) {
        // ID of the node being acted on.

        double sumForce_x = 0.0;
        double sumForce_y = 0.0;
        double sumForce_z = 0.0;

        double posA_x = posVec_x[idA];
        double posA_y = posVec_y[idA];
        double posA_z = posVec_z[idA];

        // Current node degree.
        unsigned nodeDegree = nodeDegreeVec[idA];

        for (unsigned i = 0; i < totalNodeCount; ++i) {

            double distAB_x = posVec_x[idB] - posA_x;
            double distAB_y = posVec_y[idB] - posA_y;
            double distAB_z = posVec_z[idB] - posA_z;

            double dist = norm(distAB_x, distAB_y, distAB_z);

            if (fabs(dist)>=1.0e-12) {
                //Calculate force from linear spring (Hooke's Law) on node.
                sumForce_x += LJForceByCoord(dist, distAB_x);
                sumForce_y += LJForceByCoord(dist, distAB_y);
                sumForce_z += LJForceByCoord(dist, distAB_z);            
            }
        }

        if(isfinite(sumForce_x)) {
            forceVec_x[idA] += sumForce_x;
        }
        if(isfinite(sumForce_x)) {
            forceVec_y[idA] += sumForce_y;
        }
        if(isfinite(sumForce_x)) {
            forceVec_z[idA] += sumForce_z;
        }    
        
    } // End operator()
}; // End struct

#endif