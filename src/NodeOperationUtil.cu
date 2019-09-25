#include "NodeOperationUtil.h"

#include <cmath>


__host__ __device__
bool isRegistered(
    unsigned* indicesBegin, 
    unsigned* indicesEnd,
    const unsigned nNodeTypes,
    const unsigned id,
    unsigned& nodeTypeOutput) {

        for(int i = 0; i < nNodeTypes; ++i) {
            unsigned begin = indicesBegin[i];
            unsigned end = indicesEnd[i];

            if (id >= begin) {
                if (id < end) {
                    nodeTypeOutput = i;
                    return true;
                }
            }            
        }
        return false;
}

/*
__host__ __device__
bool isRegistered(
    unsigned* indicesBegin, 
    unsigned* indicesEnd,
    const unsigned nNodeTypes,
    const unsigned id) {

        for(int i = 0; i < nNodeTypes; ++i) {
            unsigned begin = indicesBegin[i];
            unsigned end = indicesEnd[i];

            if (id >= begin) {
                if (id < end) {
                    return true;
                }
            }            
        }
        return false;
}
 */

template <typename T>
__host__ __device__ T norm(T x, T y, T z) {
        return sqrt(x*x + y*y + z*z);
}
