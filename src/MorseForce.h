#ifndef MORSE_FORCE_H_
#define MORSE_FORCE_H_


#include "IFunction.h"
#include "NodeOperation.h"


class BucketScheme;


class MorseForce : public NodeOperation, public IFunction {

    private:
    BucketScheme& bucketScheme;

    // parameters
    thrust::device_vector<double> U;
    thrust::device_vector<double> P;
    thrust::device_vector<double> R_eq;

    double energy{ 0.0 };

    public:

    MorseForce(NodeData& _nodeData, BucketScheme& _bucketScheme);

    void getDefaultParameterValues(ParameterManager& parameterManager);


    void execute();
};

#endif