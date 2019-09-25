#include "NodeData.h"

#include <thrust/tuple.h>
#include <thrust/zip_iterator.h>


NodeData::getIteratorPositionBegin() {
    return thrust::zip_iterator(
        thrust::make_tuple(
            pos_x.begin(),
            pos_y.begin(),
            pos_z.begin()));
}

NodeData::getIteratorPositionEnd() {
    return thrust::zip_iterator(
        thrust::make_tuple(
            pos_x.end(),
            pos_y.end(),
            pos_z.end()));
}