#ifndef SET_NODE_SLICE_H_
#define SET_NODE_SLICE_H_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "INodePredicate.h"

class SetNodeSlice : public INodePredicate {

    private:
    thrust::host_vector<double>::iterator begin; 
    thrust::host_vector<double>::iterator end;
    bool isTop{ true };
    double percentSlice{ 0.10 };

    public:

    SetNodeSlice(
        thrust::host_vector<double>::iterator _begin, 
        thrust::host_vector<double>::iterator _end,
        double _percentSlice,
        bool _isTop);

    void setIsTop(bool _isTop);

    void getNodeID(thrust::device_vector<unsigned>& nodeID);
};

#endif // SET_NODE_SLICE_H_