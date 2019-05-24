#ifndef FUNCTOR_LJ_FORCE_H_
#define FUNCTOR_LJ_FORCE_H_

#include "SystemStructures.h"


/* Input:   node ID */  

/* Output:  VOID */

/* Update:  node.force_(x,y,z) on given node based on LJ interactions with ALL nodes. */

// This functor does not account for fixed nodes.




__host__ __device__ double LJForceByCoord(double dist, double coordDist, 
        double U, double K, double W, double G, double L) {

    return ( (U/K) * exp(-(dist - L)/K) - (W/G) * exp(-(dist-L)/G) ) * (coordDist/dist);
}

template <typename T>
__host__ __device__ T norm(T x, T y, T z) {
        return sqrt(x*x + y*y + z*z);
}


struct functor_LJ_force : public thrust::unary_function<unsigned, void> {
    double* vec_memPos_x;
    double* vec_memPos_y;
    double* vec_memPos_z;

    double* vec_memForce_x;
    double* vec_memForce_y;
    double* vec_memForce_z;

    bool* vec_isFixed;

    unsigned memNodeCount;

    double* vec_intPos_x;
    double* vec_intPos_y;
    double* vec_intPos_z;

    double* vec_intForce_x;
    double* vec_intForce_y;
    double* vec_intForce_z;

    unsigned intNodeCount;

    double U_II;
    double K_II;
    double W_II;
    double G_II;
    double L_II;
    double U_MI;
    double K_MI;
    double W_MI;
    double G_MI;
    double L_MI;


    __host__ __device__
        functor_LJ_force(
            double* _vec_memPos_x,
            double* _vec_memPos_y,
            double* _vec_memPos_z,
            
            double* _vec_memForce_x,
            double* _vec_memForce_y,
            double* _vec_memForce_z,

            bool* _vec_isFixed,

            unsigned& _memNodeCount,

            double* _vec_intPos_x,
            double* _vec_intPos_y,
            double* _vec_intPos_z,
            
            double* _vec_intForce_x,
            double* _vec_intForce_y,
            double* _vec_intForce_z,

            unsigned& _intNodeCount,

            double _U_II,
            double _K_II,
            double _W_II,
            double _G_II,
            double _L_II,
            double _U_MI,
            double _K_MI,
            double _W_MI,
            double _G_MI,
            double _L_MI ) :

        vec_memPos_x(_vec_memPos_x),
        vec_memPos_y(_vec_memPos_y),
        vec_memPos_z(_vec_memPos_z),
        
        vec_memForce_x(_vec_memForce_x),
        vec_memForce_y(_vec_memForce_y),
        vec_memForce_z(_vec_memForce_z),

        vec_isFixed(_vec_isFixed),

        memNodeCount(_memNodeCount),

        vec_intPos_x(_vec_intPos_x),
        vec_intPos_y(_vec_intPos_y),
        vec_intPos_z(_vec_intPos_z),
        
        vec_intForce_x(_vec_intForce_x),
        vec_intForce_y(_vec_intForce_y),
        vec_intForce_z(_vec_intForce_z),

        intNodeCount(_intNodeCount),

        U_II(_U_II),
        K_II(_K_II),
        W_II(_W_II),
        G_II(_G_II),
        L_II(_L_II),
        U_MI(_U_MI),
        K_MI(_K_MI),
        W_MI(_W_MI),
        G_MI(_G_MI),
        L_MI(_L_MI) {}


    __device__
    void operator()(const unsigned id) {
        // ID of the node being acted on.
        // Membrane nodes first, then internal nodes.

        double posA_x;
        double posA_y;
        double posA_z;

        unsigned indexBegin;
        unsigned indexEnd = memNodeCount + intNodeCount;


        // ********************************************
        // Set parameters as MI.
        // These will be the first values no matter what.
        double U = U_MI;
        double K = K_MI;
        double W = W_MI;
        double G = G_MI;
        double L = L_MI;

        // ********************************************

        double* vec_force_x;
        double* vec_force_y;
        double* vec_force_z;

        bool isFixed{ false };
        bool isMemNode = (id < memNodeCount) ? true : false;

        unsigned idA;
        if (isMemNode) {
            idA = id;
            posA_x = vec_memPos_x[idA];
            posA_y = vec_memPos_y[idA];
            posA_z = vec_memPos_z[idA];

            vec_force_x = vec_memForce_x;
            vec_force_y = vec_memForce_y;
            vec_force_z = vec_memForce_z;

            isFixed = vec_isFixed[idA];

            // Don't need LJ force from other membrane nodes.
            indexBegin = memNodeCount;
        } else {
            // The ID has moved passed the membrane nodes and is now into internal nodes.
            idA = id - memNodeCount;
            posA_x = vec_intPos_x[idA];
            posA_y = vec_intPos_y[idA];
            posA_z = vec_intPos_z[idA];

            // Need LJ force from all membrane and other internal nodes.
            indexBegin = 0;

            vec_force_x = vec_intForce_x;
            vec_force_y = vec_intForce_y;
            vec_force_z = vec_intForce_z;
        }

        double sumForce_x = 0.0;
        double sumForce_y = 0.0;
        double sumForce_z = 0.0;
        if (!isFixed) {
            for (unsigned i = indexBegin; i < indexEnd; ++i) {

                double distAB_x;
                double distAB_y;
                double distAB_z;

                unsigned idB;
                if (i < memNodeCount) { // MI, since idA must be internal node.
                    // Parameters stay at MI.
                    idB = i;
                    distAB_x = vec_memPos_x[idB] - posA_x;
                    distAB_y = vec_memPos_y[idB] - posA_y;
                    distAB_z = vec_memPos_z[idB] - posA_z;
                } else { // MI or II
                    if (!isMemNode) {
                        // Change parameters to II.
                        U = U_II;
                        K = K_II;
                        W = W_II;
                        G = G_II;
                        L = L_II;
                    }

                    idB = i - memNodeCount;
                    distAB_x = vec_intPos_x[idB] - posA_x;
                    distAB_y = vec_intPos_y[idB] - posA_y;
                    distAB_z = vec_intPos_z[idB] - posA_z;
                }

                double dist = norm(distAB_x, distAB_y, distAB_z);

                if (fabs(dist)>=1.0e-12) {
                    //Calculate force from LJ potential.
                    sumForce_x += LJForceByCoord(dist, distAB_x, U, K, W, G, L);
                    sumForce_y += LJForceByCoord(dist, distAB_y, U, K, W, G, L);
                    sumForce_z += LJForceByCoord(dist, distAB_z, U, K, W, G, L);            
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
        
    } // End operator()
}; // End struct

#endif