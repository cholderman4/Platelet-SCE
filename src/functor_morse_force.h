#ifndef FUNCTOR_MORSE_FORCE_H_
#define FUNCTOR_MORSE_FORCE_H_

#include <cmath>

#include <thrust/tuple.h>

#include "NodeOperationUtil.h"


// Input:   node ID, bucket ID  

// Output:  energy 

// Update:  node.force_(x,y,z) on given node based on LJ interactions with ALL nodes.


typedef thrust::tuple<unsigned, unsigned> Tuu;


__host__ __device__ 
double morseForceByCoord(double R, double coordDist, 
        double U, double P, double R_eq) {


    // New formula for Morse potential.
    // Newman 2008
    
    return 4 * U * P * (R/R_eq) 
        * ( exp( 2 * P * (1 - (R/R_eq)*(R/R_eq))) - exp( P * (1 - (R/R_eq)*(R/R_eq))) )
        * coordDist/R;
}

__host__ __device__ 
double morseEnergy(double R, double U, double P, double R_eq) {

    return 0.5*U*( exp( 2*P*(1 - (R/R_eq)*(R/R_eq))) - 2*exp(P*(1 - (R/R_eq)*(R/R_eq))) ); 
}


struct functor_morse_force : public thrust::unary_function<Tuu, double> {
    double* vec_pos_x;
    double* vec_pos_y;
    double* vec_pos_z;

    double* vec_force_x;
    double* vec_force_y;
    double* vec_force_z;

    unsigned* vec_indicesBegin;
    unsigned* vec_indicesEnd;
    unsigned nNodeTypes;

    unsigned* globalNode_ID_expanded;
	unsigned* keyBegin;
	unsigned* keyEnd;

    double* vec_U;
    double* vec_P;
    double* vec_R_eq;


    __host__ __device__
        functor_morse_force(
            double* _vec_pos_x,
            double* _vec_pos_y,
            double* _vec_pos_z,
            
            double* _vec_force_x,
            double* _vec_force_y,
            double* _vec_force_z,

            unsigned* _vec_indicesBegin,
            unsigned* _vec_indicesEnd,
            unsigned _nNodeTypes,

            unsigned* _globalNode_ID_expanded,
            unsigned* _keyBegin,
            unsigned* _keyEnd,

            double* _vec_U,
            double* _vec_P,
            double* _vec_R_eq ) :

        vec_pos_x(_vec_pos_x),
        vec_pos_y(_vec_pos_y),
        vec_pos_z(_vec_pos_z),
        
        vec_force_x(_vec_force_x),
        vec_force_y(_vec_force_y),
        vec_force_z(_vec_force_z),

        vec_indicesBegin(_vec_indicesBegin),
        vec_indicesEnd(_vec_indicesEnd),
        nNodeTypes(_nNodeTypes),

        globalNode_ID_expanded(_globalNode_ID_expanded),
		keyBegin(_keyBegin),
		keyEnd(_keyEnd),

        vec_U(_vec_U),
        vec_P(_vec_P),
        vec_R_eq(_vec_R_eq) {}


    __device__
    double operator()(const Tuu& u2) {
        
        // ID of the node being acted on.
        unsigned idA = thrust::get<0>(u2);

        // Bucket that current node is in. 
        unsigned bucket_ID = thrust::get<1>(u2);

        double posA_x = vec_pos_x[idA];
        double posA_y = vec_pos_y[idA];
        double posA_z = vec_pos_z[idA];

        unsigned indexBegin = keyBegin[bucket_ID];
        unsigned indexEnd = keyEnd[bucket_ID];        

        double sumForce_x = 0.0;
        double sumForce_y = 0.0;
        double sumForce_z = 0.0;

        double sumEnergy{ 0.0 };

        unsigned nodeTypeA;

        if ( isRegistered(
                vec_indicesBegin, vec_indicesEnd, 
                nNodeTypes, idA,
                nodeTypeA) ) {

            for (unsigned globalNodeIndex = indexBegin; globalNodeIndex < indexEnd; ++globalNodeIndex) {

                unsigned idB = globalNode_ID_expanded[globalNodeIndex];

                unsigned nodeTypeB;
                
                if (isRegistered(
                vec_indicesBegin, vec_indicesEnd, 
                nNodeTypes, idB,
                nodeTypeB) ) {

                    unsigned parameterIndex = nodeTypeA * nNodeTypes + nodeTypeB;
                    double U = vec_U[parameterIndex];
                    double P = vec_P[parameterIndex];
                    double R_eq = vec_R_eq[parameterIndex];

                    double distAB_x = vec_pos_x[idB] - posA_x;
                    double distAB_y = vec_pos_y[idB] - posA_y;
                    double distAB_z = vec_pos_z[idB] - posA_z;

                    double dist = norm(distAB_x, distAB_y, distAB_z);
                    double tempEnergy = morseEnergy(dist, U, P, R_eq);
                    if (isfinite(tempEnergy)) {
                        sumEnergy += tempEnergy;
                    }

                    if (fabs(dist)>=1.0e-12) {
                    // if(true) {
                        //Calculate force from Morse potential.
                        sumForce_x += morseForceByCoord(dist, distAB_x, U, P, R_eq);
                        sumForce_y += morseForceByCoord(dist, distAB_y, U, P, R_eq);
                        sumForce_z += morseForceByCoord(dist, distAB_z, U, P, R_eq);            
                    }

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