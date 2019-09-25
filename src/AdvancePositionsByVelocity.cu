#include "AdvancePositionsByVelocity.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "functor_advance_by_vel.h"
#include "NodeData.h"
#include "NodeOperation.h"
#include "NodeOperationUtil.h"

AdvancePositionsByVelocity::AdvancePositionsByVelocity(NodeData& _nodeData) :
    NodeOperation(_nodeData) {};

void AdvancePositionsByVelocity::SetVelocity(
    double _vel_x, 
    double _vel_y, 
    double _vel_z) {
        vel_x = _vel_x;
        vel_y = _vel_y;
        vel_z = _vel_z;
    };

void AdvancePositionsByVelocity::SetDirectionAndMagnitude(
    double _dir_x, 
    double _dir_y, 
    double _dir_z, 
    double _magnitude) {

        double dist = norm(_dir_x, _dir_y, _dir_z);

        if (dist != 0.0) {
            vel_x = _dir_x * _magnitude / dist;
            vel_y = _dir_y * _magnitude / dist;
            vel_z = _dir_z * _magnitude / dist;
        } else {
            vel_x = 0.0;
            vel_y = 0.0;
            vel_z = 0.0;
        }
};

void AdvancePositionsByVelocity::SetTimeStep(double _dt) {
    dt = _dt;
}

void AdvancePositionsByVelocity::execute() {

    thrust::counting_iterator<unsigned> iteratorStart(0);
    
    // unsigned nNodesTransformed = 0;

    for (int i = 0; i < nNodeTypes; ++i) {
        // Each for loop iteration corresponds to a new continuous chunk of data.
        unsigned begin = indicesBegin[i];

        // Find the end of the chunk.
        bool isConnected = true;
        while ( isConnected ) {
            if (i+1 < nNodeTypes) {
                if( indicesEnd[i] == indicesBegin[i+1] ) {
                    ++i;
                } else {
                    isConnected = false;
                } 
            } else {
                isConnected = false;
            }
        }
        unsigned end = indicesEnd[i];

        thrust::transform(
            // Input vector #1
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    iteratorStart,
                    nodeData.pos_x.begin(),
                    nodeData.pos_y.begin(),
                    nodeData.pos_z.begin())) + begin,
    
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    iteratorStart,
                    nodeData.pos_x.begin(),
                    nodeData.pos_y.begin(),
                    nodeData.pos_z.begin())) + end,
    
            // Output vector
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    nodeData.pos_x.begin(),
                    nodeData.pos_y.begin(),
                    nodeData.pos_z.begin())) + begin,
    
            // Functor + parameter call
            functor_advance_by_vel(   
                dt,
                vel_x,
                vel_y,
                vel_z));

        // nNodesTransformed += (end - begin);
    }
}
