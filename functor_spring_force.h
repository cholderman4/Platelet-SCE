#ifndef FUNCTOR_SPRING_FORCE_H_
#define FUNCTOR_SPRING_FORCE_H_

#include "SystemStructures.h"


/* Input:   node ID */  

/* Output:  VOID */

/* Update:  node.force_(x,y,z) on given node based on all spring connections. */

// This functor does not account for fixed nodes.




__host__ __device__ double springForceByCoord(double dist, double coordDist, double l_0) {
    double k = 2.1;
    // double eq = 3.0;

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

    double* len_0;
    unsigned* nodeConnections;

    unsigned maxConnectedSpringCount;


    __host__ __device__
        functor_spring_force(
            double* _posVec_x,
            double* _posVec_y,
            double* _posVec_z,
            
            double* _forceVec_x,
            double* _forceVec_y,
            double* _forceVec_z,

            double* _len_0,
            unsigned* _nodeConnections,

            unsigned _maxConnectedSpringCount) :

        posVec_x(_posVec_x),
        posVec_y(_posVec_y),
        posVec_z(_posVec_z),
        
        forceVec_x(_forceVec_x),
        forceVec_y(_forceVec_y),
        forceVec_z(_forceVec_z),

        len_0(_len_0),
        nodeConnections(_nodeConnections),

        maxConnectedSpringCount(_maxConnectedSpringCount) {}


    __device__
    void operator()(const unsigned idA) {
        // ID of the node being acted on.

        double sumForce_x = 0.0;
        double sumForce_y = 0.0;
        double sumForce_z = 0.0;

        double posA_x = posVec_x[idA];
        double posA_y = posVec_y[idA];
        double posA_z = posVec_z[idA];

        // Index along nodeConnections vector.
        unsigned indexBegin = idA * maxConnectedSpringCount;
        unsigned indexEnd = indexBegin + maxConnectedSpringCount;

        for (unsigned i=indexBegin; i < indexEnd; ++i) {

            // ID of node(s) connected to the primary node.
            unsigned idB = nodeConnections[i];

            double distAB_x = posA_x - posVec_x[idB];
            double distAB_y = posA_y - posVec_y[idB];
            double distAB_z = posA_z - posVec_z[idB];

            double dist = norm(distAB_x, distAB_y, distAB_z);

            double length_0 = len_0[i];

            if (fabs(dist)>=1.0e-12) {
                //Calculate force from linear spring (Hooke's Law) on node.
                sumForce_x += springForceByCoord(dist, distAB_x, length_0);
                sumForce_y += springForceByCoord(dist, distAB_y, length_0);
                sumForce_z += springForceByCoord(dist, distAB_z, length_0);            
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