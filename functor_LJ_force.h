#ifndef FUNCTOR_LJ_FORCE_H_
#define FUNCTOR_LJ_FORCE_H_

#include "SystemStructures.h"


/* Input:   node ID */  

/* Output:  VOID */

/* Update:  node.force_(x,y,z) on given node based on LJ interactions with ALL nodes. */

// This functor does not account for fixed nodes.




__host__ __device__ 
double LJForceByCoord(double R, double coordDist, 
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

template <typename T>
__host__ __device__ T norm(T x, T y, T z) {
        return sqrt(x*x + y*y + z*z);
}


struct functor_LJ_force : public thrust::unary_function<Tuu, double> {
    double* vec_pos_x;
    double* vec_pos_y;
    double* vec_pos_z;

    double* vec_force_x;
    double* vec_force_y;
    double* vec_force_z;

    bool* vec_isFixed;

    unsigned memNodeCount;
    unsigned intNodeCount;

    unsigned* globalNode_ID_expanded;
	unsigned* keyBegin;
	unsigned* keyEnd;

    double U_II;
    double P_II;
    double R_eq_II;

    double U_MI;
    double P_MI;
    double R_eq_MI;


    __host__ __device__
        functor_LJ_force(
            double* _vec_pos_x,
            double* _vec_pos_y,
            double* _vec_pos_z,
            
            double* _vec_force_x,
            double* _vec_force_y,
            double* _vec_force_z,

            bool* _vec_isFixed,

            unsigned& _memNodeCount,
            unsigned& _intNodeCount,

            unsigned* _globalNode_ID_expanded,
            unsigned* _keyBegin,
            unsigned* _keyEnd,

            double _U_II,
            double _P_II,
            double _R_eq_II,

            double _U_MI,
            double _P_MI,
            double _R_eq_MI ) :

        vec_pos_x(_vec_pos_x),
        vec_pos_y(_vec_pos_y),
        vec_pos_z(_vec_pos_z),
        
        vec_force_x(_vec_force_x),
        vec_force_y(_vec_force_y),
        vec_force_z(_vec_force_z),

        vec_isFixed(_vec_isFixed),

        memNodeCount(_memNodeCount),
        intNodeCount(_intNodeCount),

        globalNode_ID_expanded(_globalNode_ID_expanded),
		keyBegin(_keyBegin),
		keyEnd(_keyEnd),

        U_II(_U_II),
        P_II(_P_II),
        R_eq_II(_R_eq_II),

        U_MI(_U_MI),
        P_MI(_P_MI),
        R_eq_MI(_R_eq_MI) {}


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

        bool isFixed{ false };
        bool isMemNode = (idA < memNodeCount) ? true : false;

        if (isMemNode) {
            isFixed = vec_isFixed[idA];
        } 

        double sumForce_x = 0.0;
        double sumForce_y = 0.0;
        double sumForce_z = 0.0;

        double sumEnergy{ 0.0 };

        if (!isFixed) {
            for (unsigned globalNodeIndex = indexBegin; globalNodeIndex < indexEnd; ++globalNodeIndex) {

                unsigned idB = globalNode_ID_expanded[globalNodeIndex];

                double U;
                double P;
                double R_eq;
                if ( (idB < memNodeCount) && isMemNode ) { 
                    // MM interaction, no LJ force
                    continue;    
                } else if ( (idB >= memNodeCount) && !isMemNode ) { 
                    // Change parameters to II.
                    U = U_II;
                    P = P_II;
                    R_eq = R_eq_II;
                } else {
                    U = U_MI;
                    P = P_MI;
                    R_eq = R_eq_MI;
                }

                double distAB_x = vec_pos_x[idB] - posA_x;
                double distAB_y = vec_pos_y[idB] - posA_y;
                double distAB_z = vec_pos_z[idB] - posA_z;

                double dist = norm(distAB_x, distAB_y, distAB_z);
                double tempEnergy = morseEnergy(dist, U, P, R_eq);
                if (isfinite(tempEnergy)) {
                    sumEnergy += tempEnergy;
                }

                if (fabs(dist)>=1.0e-12) {
                    //Calculate force from LJ potential.
                    sumForce_x += LJForceByCoord(dist, distAB_x, U, P, R_eq);
                    sumForce_y += LJForceByCoord(dist, distAB_y, U, P, R_eq);
                    sumForce_z += LJForceByCoord(dist, distAB_z, U, P, R_eq);            
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