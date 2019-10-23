#ifndef MORSE_FORCE_H_
#define MORSE_FORCE_H_


#include "IFunction.h"
#include "SymmetricForce.h"


class BucketScheme;


class MorseForce : public SymmetricForce, public IFunction {

    private:
    BucketScheme& bucketScheme;

    // parameters
    thrust::device_vector<double> U;
    thrust::device_vector<double> P;
    thrust::device_vector<double> R_eq;

    double energy{ 0.0 };

    public:

    MorseForce(NodeData& _nodeData, BucketScheme& _bucketScheme);

    void getParameterKeys(ParameterManager& parameterManager);
    void setParameterValues(ParameterManager& parameterManager);


    void execute();
};

#endif