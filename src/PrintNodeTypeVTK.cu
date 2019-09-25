#include "PrintNodeTypeVTK.h"

#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>

#include "functor_output_file.h"
#include "NodeData.h"

PrintNodeTypeVTK::PrintNodeTypeVTK(NodeData& _nodeData) :
    nodeData(_nodeData) {}


void PrintNodeTypeVTK::print(std::ofstream& ofs) {

    unsigned nNodesTotal = nodeData.getNodeCount();

    ofs << "POINT_DATA " << nNodesTotal << std::endl;
    ofs << "SCALARS NodeType FLOAT \n";
    ofs << "LOOKUP_TABLE default \n";

    unsigned nNodeTypes = nodeData.getNodeTypeCount();
    for (unsigned i = 0; i < nNodeTypes; ++i) {

        thrust::constant_iterator<unsigned> nodeType(i);
        unsigned nNodes = nodeData.getIndexEnd(i) - nodeData.getIndexBegin(i);
        
        thrust::for_each(
            nodeType, nodeType + nNodes,
            functor_output_value<unsigned>(ofs));        
    }
    
}
