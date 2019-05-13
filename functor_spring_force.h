#ifndef FUNCTOR_SPRING_FORCE_H_
#define FUNCTOR_SPRING_FORCE_H_

#include "SystemStructures.h"


/* Input:   node ID */  

/* Output:  VOID */

/* Update:  node.force_(x,y,z) on given node based on all spring connections. */

// This functor does not account for fixed nodes.
// (Perhaps it should.)




__host__ __device__ double springForceByCoord(double dist, double coordDist, double k, double l_0) {
    // double k = 20;

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

    unsigned* nodeIDVec_L;
    unsigned* nodeIDVec_R;
    double* len_0;
    unsigned* nodeConnections;
    unsigned* nodeDegreeVec;

    unsigned maxConnectedSpringCount;
    double memSpringStiffness;


    __host__ __device__
        functor_spring_force(
            double* _posVec_x,
            double* _posVec_y,
            double* _posVec_z,
            
            double* _forceVec_x,
            double* _forceVec_y,
            double* _forceVec_z,

            unsigned* _nodeIDVec_L,
            unsigned* _nodeIDVec_R,
            double* _len_0,
            unsigned* _nodeConnections,
            unsigned* _nodeDegreeVec,

            unsigned& _maxConnectedSpringCount,
            double & _memSpringStiffness) :

        posVec_x(_posVec_x),
        posVec_y(_posVec_y),
        posVec_z(_posVec_z),
        
        forceVec_x(_forceVec_x),
        forceVec_y(_forceVec_y),
        forceVec_z(_forceVec_z),

        nodeIDVec_L(_nodeIDVec_L),
        nodeIDVec_R(_nodeIDVec_R),
        len_0(_len_0),
        nodeConnections(_nodeConnections),
        nodeDegreeVec(_nodeDegreeVec),

        maxConnectedSpringCount(_maxConnectedSpringCount),
        memSpringStiffness(_memSpringStiffness) {}


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

        // Index along nodeConnections vector.
        unsigned indexBegin = idA * maxConnectedSpringCount;
        unsigned indexEnd = indexBegin + nodeDegree;

        for (unsigned i = indexBegin; i < indexEnd; ++i) {

            unsigned springID = nodeConnections[i];
            // double length_0 = len_0[springID];
            double length_0 = 0.060;

            // Temporary test value.
            // double length_0 = 0.50;

            // ID of node(s) connected to the primary node.
            unsigned idB;
            if ( nodeIDVec_L[springID] == idA) {
                idB = nodeIDVec_R[springID];
            } else {
                idB = nodeIDVec_L[springID];
            }

            double distAB_x = posVec_x[idB] - posA_x;
            double distAB_y = posVec_y[idB] - posA_y;
            double distAB_z = posVec_z[idB] - posA_z;

            double dist = norm(distAB_x, distAB_y, distAB_z);

            if (fabs(dist)>=1.0e-12) {
                //Calculate force from linear spring (Hooke's Law) on node.
                sumForce_x += springForceByCoord(dist, distAB_x, memSpringStiffness, length_0);
                sumForce_y += springForceByCoord(dist, distAB_y, memSpringStiffness, length_0);
                sumForce_z += springForceByCoord(dist, distAB_z, memSpringStiffness, length_0);            
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