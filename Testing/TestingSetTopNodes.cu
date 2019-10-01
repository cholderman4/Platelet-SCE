#include "SetTopNodes.h"

#include <iostream>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


int main() {

    thrust::host_vector<double> data(10);
    data[0] = 0.1;
    data[1] = 10.4;
    data[2] = 100.3;
    data[3] = -20.5;
    data[4] = 12.3;
    data[5] = 90.2;
    data[6] = 30.7;
    data[7] = 50.0;
    data[8] = 78;
    data[9] = 65.222;

    SetTopNodes topNodes(data.begin(), data.end(), 0.30, true);

    thrust::device_vector<unsigned> nodeID;

    topNodes.getNodeID(nodeID);

    thrust::copy(
        nodeID.begin(), 
        nodeID.end(), 
        std::ostream_iterator<double>(std::cout, " "));


    return 0;
}