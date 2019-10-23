#include "PrintNodePositionsVTK.h"

#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>

#include "functor_output_file.h"
#include "NodeData.h"

PrintNodePositionsVTK::PrintNodePositionsVTK(NodeData& _nodeData) :
    nodeData(_nodeData) {}


void PrintNodePositionsVTK::print(std::ofstream& ofs) {

    unsigned nNodes = nodeData.getNodeCount();

    thrust::host_vector<double> hostPos_x(nNodes);
    thrust::host_vector<double> hostPos_y(nNodes);
    thrust::host_vector<double> hostPos_z(nNodes);

/* 
    hostPos_x = nodeData.pos_x;
    hostPos_y = nodeData.pos_y;
    hostPos_z = nodeData.pos_z;
 */
 
    thrust::copy(nodeData.pos_x.begin(), nodeData.pos_x.end(), hostPos_x.begin());
    thrust::copy(nodeData.pos_y.begin(), nodeData.pos_y.end(), hostPos_y.begin());
    thrust::copy(nodeData.pos_z.begin(), nodeData.pos_z.end(), hostPos_z.begin());    

    ofs << "POINTS " << nNodes << " FLOAT" << std::endl;

    thrust::for_each(
        thrust::make_zip_iterator(
            thrust::make_tuple(
                hostPos_x.begin(),
                hostPos_y.begin(),
                hostPos_z.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                hostPos_x.end(),
                hostPos_y.end(),
                hostPos_z.end())),
        functor_output_tuple(ofs));
}
