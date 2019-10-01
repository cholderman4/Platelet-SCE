#include "SetNodeSlice.h"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>

#include "functor_slice_predicate.h"


SetNodeSlice::SetNodeSlice(
    thrust::host_vector<double>::iterator _begin, 
    thrust::host_vector<double>::iterator _end, 
    double _percentSlice,
    bool _isTop) : 

    begin(_begin),
    end(_end) {
        percentSlice = _percentSlice;
        setIsTop(_isTop);
     };


void SetNodeSlice::setIsTop(bool _isTop) {
    isTop = _isTop;
}


void SetNodeSlice::getNodeID(thrust::device_vector<unsigned>& nodeID) {
    
    unsigned N = end - begin;
    thrust::host_vector<unsigned> hostNodeID(N);

    auto max = *(thrust::max_element(begin, end));
    auto min = *(thrust::min_element(begin, end));

    double range = max - min;

    double threshold;
    if (isTop == true) {
        threshold = max - (percentSlice * range);
    } else {
        threshold = min + (percentSlice * range);
    }

    thrust::counting_iterator<unsigned> iteratorStart(0);

    auto endReduced = thrust::copy_if(
        iteratorStart,
        iteratorStart + N,
        begin,
        hostNodeID.begin(), 
        functor_slice_predicate(threshold, isTop));

    unsigned reducedSize = endReduced - hostNodeID.begin();
    hostNodeID.resize(reducedSize);
    nodeID.resize(reducedSize);
    thrust::copy(hostNodeID.begin(), hostNodeID.end(), nodeID.begin());
}
