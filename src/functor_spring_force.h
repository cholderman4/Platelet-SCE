#ifndef FUNCTOR_SPRING_FORCE_H_
#define FUNCTOR_SPRING_FORCE_H_

#include <cmath>

#include "NodeOperationUtil.h"


/* Input:   node ID */  

/* Output:  VOID */

/* Update:  memNode.force_(x,y,z) on given node based on all spring connections. */

// This functor does not account for fixed nodes.




__host__ __device__ 
double springForceByCoord(double dist, double coordDist, double k, double l_0) {
    // double k = 20;

    return k*(dist - l_0)*coordDist/dist;
}

__host__ __device__
double springEnergy(double R, double K, double l_0) {

    return 0.5*K*(R - l_0)*(R - l_0);
}


struct functor_spring_force : public thrust::unary_function<unsigned, double> {
    double* vec_pos_x;
    double* vec_pos_y;
    double* vec_pos_z;

    double* vec_force_x;
    double* vec_force_y;
    double* vec_force_z;

    unsigned* vec_springConnectionsByNode;
    unsigned* vec_nSpringsConnectedToNode;
    unsigned nMaxSpringsConnectedToNode;
    unsigned nNodesTransformed;

    unsigned* vec_nodeConnectionsBySpring;
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

            unsigned* _vec_springConnectionsByNode,
            unsigned* _vec_nSpringsConnectedToNode,
            unsigned& _nMaxSpringsConnectedToNode,
            unsigned& _nNodesTransformed,

            unsigned* _vec_nodeConnectionsBySpring,
            double* _vec_len_0,
            double& _memSpringStiffness) :

        vec_pos_x(_vec_pos_x),
        vec_pos_y(_vec_pos_y),
        vec_pos_z(_vec_pos_z),
        
        vec_force_x(_vec_force_x),
        vec_force_y(_vec_force_y),
        vec_force_z(_vec_force_z),

        vec_springConnectionsByNode(_vec_springConnectionsByNode),
        vec_nSpringsConnectedToNode(_vec_nSpringsConnectedToNode),
        nMaxSpringsConnectedToNode(_nMaxSpringsConnectedToNode),
        nNodesTransformed(_nNodesTransformed),

        vec_nodeConnectionsBySpring(_vec_nodeConnectionsBySpring),
        vec_len_0(_vec_len_0),
        memSpringStiffness(_memSpringStiffness) {}


    __device__
    double operator()(const unsigned& idA) {
        // ID of the node being acted on.
        // unsigned idA = thrust::get<0>(u1b1);
        // bool isFixed = thrust::get<1>(u1b1);

        double sumEnergy{ 0.0 };
        
        double sumForce_x{ 0.0 };
        double sumForce_y{ 0.0 };
        double sumForce_z{ 0.0 };

        double posA_x = vec_pos_x[idA];
        double posA_y = vec_pos_y[idA];
        double posA_z = vec_pos_z[idA];

        // Current node degree.
        unsigned nConnectedSprings = vec_nSpringsConnectedToNode[idA];

        // Index along vec_springConnectionsByNode vector.
        unsigned indexBegin = (idA - nNodesTransformed) * nMaxSpringsConnectedToNode;
        unsigned indexEnd = indexBegin + nConnectedSprings;

        for (unsigned i = indexBegin; i < indexEnd; ++i) {

            // ***************************************************
            // Getting parameters.
            //  len_0:  different for every spring, use a vector.
            // stiffness: constant
            unsigned connectedSpringID = vec_springConnectionsByNode[i];
            double length_0 = vec_len_0[connectedSpringID];

            // ID of node(s) connected to the primary node.
            unsigned idB;
            if ( vec_nodeConnectionsBySpring[2 * connectedSpringID] == idA) {
                idB = vec_nodeConnectionsBySpring[2*connectedSpringID + 1];
            } else {
                idB = vec_nodeConnectionsBySpring[2 * connectedSpringID];
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


        return sumEnergy;            
        
    } // End operator()
}; // End struct

#endif