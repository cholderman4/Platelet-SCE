#ifndef RANDOM_UTIL_H_
#define RANDOM_UTIL_H_

#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>


///////////////////////////////////////////////
//random number generators
struct psrnormgen {

    double a, b;

    __host__ __device__
	psrnormgen(
		double _a,
		double _b) :
		a(_a),
		b(_b) {}

    __host__ __device__ 
	double operator()(const unsigned n) const
    {
        thrust::default_random_engine rng(n);
        thrust::normal_distribution<float> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }

};

struct psrunifgen {

    double a, b;

    __host__ __device__
	psrunifgen(
		double _a,
		double _b) :
		a(_a),
		b(_b) {}

    __host__ __device__ 
	double operator()(const unsigned n) const
    {
        thrust::default_random_engine rng(n);
        thrust::uniform_real_distribution<float> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }

};

#endif