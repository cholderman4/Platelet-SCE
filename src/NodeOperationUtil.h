#ifndef NODE_OPERATION_UTIL_H_
#define NODE_OPERATION_UTIL_H_



/* __host__ __device__
bool isRegistered(
    unsigned* indexBegin, 
    unsigned* indexEnd,
    const unsigned nNodeTypes,
    const unsigned id,
    unsigned& nodeType,
    unsigned& nSkippedNodes);


__host__ __device__
bool isRegistered(
    unsigned* indexBegin, 
    unsigned* indexEnd,
    const unsigned nNodeTypes,
    const unsigned id,
    unsigned& nodeType);


__host__ __device__
bool isRegistered(
    unsigned* indexBegin, 
    unsigned* indexEnd,
    const unsigned nNodeTypes,
    const unsigned id); */


template <typename T>
__host__ __device__ T norm(T x, T y, T z);

#endif