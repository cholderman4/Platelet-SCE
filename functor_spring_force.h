#ifndef FUNCTOR_SPRING_FORCE_H_
#define FUNCTOR_SPRING_FORCE_H_

#include "SystemStructures.h"


/* Input:   node ID */  

/* Output:  VOID */

/* Update:  memNode.force_(x,y,z) on given node based on all spring connections. */

// This functor does account for fixed nodes.




__host__ __device__ 
double springForceByCoord(double dist, double coordDist, double k, double l_0) {
    // double k = 20;

    return k*(dist - l_0)*coordDist/dist;
}

__host__ __device__
double springEnergy(double R, double K, double l_0) {

    return 0.5*K*(R - l_0)*(R - l_0);
}

template <typename T>
__host__ __device__ T norm(T x, T y, T z) {
        return sqrt(x*x + y*y + z*z);
}


struct functor_spring_force : public thrust::unary_function<Tub, double> {
    double* vec_pos_x;
    double* vec_pos_y;
    double* vec_pos_z;

    double* vec_force_x;
    double* vec_force_y;
    double* vec_force_z;

    bool* vec_isFixed;
    
    unsigned* vec_connectedSpringID;
    unsigned* vec_connectedSpringCount;

    unsigned maxConnectedSpringCount; 
    
    unsigned* vec_nodeID_L;
    unsigned* vec_nodeID_R;
    double* vec_len_0;

    double memSpringStiffness;


    __host__ __device__
        functor_spring_force(
            double* _vec_pos_x,
            double* _vec_pos_y,
            double* _vec_pos_z,
            
            double* _vec_force_x,
            double* _vec_force_y,
            double* _vec_force_z,

            bool* _vec_isFixed,

            unsigned* _vec_connectedSpringID,
            unsigned* _vec_connectedSpringCount,

            unsigned& _maxConnectedSpringCount,

            unsigned* _vec_nodeID_L,
            unsigned* _vec_nodeID_R,
            double* _vec_len_0,

            double& _memSpringStiffness) :

        vec_pos_x(_vec_pos_x),
        vec_pos_y(_vec_pos_y),
        vec_pos_z(_vec_pos_z),
        
        vec_force_x(_vec_force_x),
        vec_force_y(_vec_force_y),
        vec_force_z(_vec_force_z),

        vec_isFixed(_vec_isFixed),

        vec_connectedSpringID(_vec_connectedSpringID),
        vec_connectedSpringCount(_vec_connectedSpringCount),

        maxConnectedSpringCount(_maxConnectedSpringCount),

        vec_nodeID_L(_vec_nodeID_L),
        vec_nodeID_R(_vec_nodeID_R),
        vec_len_0(_vec_len_0),

        memSpringStiffness(_memSpringStiffness) {}


    __device__
    double operator()(const Tub& u1b1) {
        // ID of the node being acted on.
        unsigned idA = thrust::get<0>(u1b1);
        bool isFixed = thrust::get<1>(u1b1);

        double sumEnergy{ 0.0 };

        if (!isFixed) {
            double sumForce_x = 0.0;
            double sumForce_y = 0.0;
            double sumForce_z = 0.0;

            double posA_x = vec_pos_x[idA];
            double posA_y = vec_pos_y[idA];
            double posA_z = vec_pos_z[idA];

            // Current node degree.
            unsigned connectedSpringCount = vec_connectedSpringCount[idA];

            // Index along vec_connectedSpringID vector.
            unsigned indexBegin = idA * maxConnectedSpringCount;
            unsigned indexEnd = indexBegin + connectedSpringCount;

            for (unsigned i = indexBegin; i < indexEnd; ++i) {

                unsigned connectedSpringID = vec_connectedSpringID[i];
                double length_0 = vec_len_0[connectedSpringID];
                // double length_0 = 0.060;

                // Temporary test value.
                // double length_0 = 0.50;

                // ID of node(s) connected to the primary node.
                unsigned idB;
                if ( vec_nodeID_L[connectedSpringID] == idA) {
                    idB = vec_nodeID_R[connectedSpringID];
                } else {
                    idB = vec_nodeID_L[connectedSpringID];
                }

                double distAB_x = vec_pos_x[idB] - posA_x;
                double distAB_y = vec_pos_y[idB] - posA_y;
                double distAB_z = vec_pos_z[idB] - posA_z;

                double dist = norm(distAB_x, distAB_y, distAB_z);

                double tempEnergy = springEnergy(dist, memSpringStiffness, length_0);
                if (isfinite(tempEnergy)){
                    sumEnergy += tempEnergy;
                }

                if (fabs(dist)>=1.0e-12) {
                    //Calculate force from linear spring (Hooke's Law) on node.
                    sumForce_x += springForceByCoord(dist, distAB_x, memSpringStiffness, length_0);
                    sumForce_y += springForceByCoord(dist, distAB_y, memSpringStiffness, length_0);
                    sumForce_z += springForceByCoord(dist, distAB_z, memSpringStiffness, length_0);            
                }
            }

            if(isfinite(sumForce_x)) {
                vec_force_x[idA] += sumForce_x;
            }
            if(isfinite(sumForce_x)) {
                vec_force_y[idA] += sumForce_y;
            }
            if(isfinite(sumForce_x)) {
                vec_force_z[idA] += sumForce_z;
            }

        }

        return sumEnergy;            
        
    } // End operator()
}; // End struct

#endif