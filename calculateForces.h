#ifndef CALCULATE_FORCES_H
#define CALCULATE_FORCES_H

#include <cmath>
#include "thrust/device_vector.h"


// Input:   bool - whether to return force to left/right node connected to edge.
//          unsigned(x2) - ID of nodes that edge is connected to.
typedef thrust::tuple<bool, unsigned, unsigned> Tbuu;

//Output:   unsigned - ID of node
//          (x,y,z) - forces on node 
typedef thrust::tuple<unsigned, double, double, double> Tuddd;


template <typename T>
__host__ __device__ T norm(T x, T y, T z) {
        return sqrt(x*x + y*y + z*z);
}


template <typename T>
__host__ __device__ T getSpringForce(T dist, T coordDist) {
    double k = 2.1;
    double eq = 3.0;

    return k*(dist - eq)*coordDist/dist;
}


struct getForcesFunctor : public thrust::unary_function<Tbuu, Tuddd> {
    double* posXVec;
    double* posYVec;
    double* posZVec;
   
    __host__ __device__
        getForcesFunctor(
            double* _posXVec,
            double* _posYVec,
            double* _posZVec) :
        posXVec(_posXVec),
        posYVec(_posYVec),
        posZVec(_posZVec) {}

    __device__
    Tuddd operator()(const Tbuu& leftNodeReturn_Id2) {
        bool leftNodeReturn = thrust::get<0>(leftNodeReturn_Id2);
        unsigned node_L = thrust::get<1>(leftNodeReturn_Id2);
        unsigned node_R = thrust::get<2>(leftNodeReturn_Id2);

        double posX_L = posXVec[node_L];
        double posY_L = posYVec[node_L];
        double posZ_L = posZVec[node_L];

        double posX_R = posXVec[node_R];
        double posY_R = posYVec[node_R];
        double posZ_R = posZVec[node_R];

        double dist = norm(posX_L - posX_R, posY_L - posY_R, posZ_L - posZ_R);

        double forceX=0.0;
        double forceY=0.0;
        double forceZ=0.0;

        if (fabs(dist)>=1.0e-16) {
            //Calculate force from spring (Hooke's Law) on node.
            forceX = getSpringForce(dist, posX_L - posX_R);
            forceY = getSpringForce(dist, posY_L - posY_R);
            forceZ = getSpringForce(dist, posZ_L - posZ_R);            
        }


        // Return the force on either left/right node.
        if (leftNodeReturn) {
            return thrust::make_tuple(node_L, forceX, forceY, forceZ);
        }
        else {
            return thrust::make_tuple(node_R, -forceX, -forceY, -forceZ);
        }
    }
};

#endif