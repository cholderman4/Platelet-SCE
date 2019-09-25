#ifndef NODE_DATA_H_
#define NODE_DATA_H_

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

enum NODE_TYPE {
    MEMBRANE,
    INTERIOR,
    EXTERIOR
};

class NodeData {
    
    // Only holds data common to every node
public:

    // ***********************************
    // Platelet data
    // Each node type will be grouped together.
    // When new nodes are added, it will be at the end of their group.
    thrust::device_vector<double> pos_x;
    thrust::device_vector<double> pos_y;
    thrust::device_vector<double> pos_z;

    thrust::device_vector<double> velocity;

    thrust::device_vector<double> force_x;
    thrust::device_vector<double> force_y;
    thrust::device_vector<double> force_z;

    thrust::device_vector<double> energy;

    // ***********************************


    // Holds the begin/end index for each node type.
    // Size is nNodeTypes + 1
    // nth nodeType has:
    //      indexBegin = nodeTypeIndex[n]
    //      indexEnd = nodeTypeIndex[n+1]
    std::vector<unsigned> nodeTypeIndex;

    // Default parameter values.
    // NodeType classes will hold more specific values, if necessary.
    double defaultMass{ 1.0 };

    // nodeType is the position in the index vector.
    // No need for this class to know anything about the types
    // other than their positions in the vector.
    unsigned getIndexBegin(const unsigned& nodeTypeID) { return nodeTypeIndex[nodeTypeID]; }
    unsigned getIndexEnd(const unsigned& nodeTypeID){ return nodeTypeIndex[nodeTypeID + 1]; }
    unsigned getNodeCount() {return pos_x.size(); }
    unsigned getNodeTypeCount() { return nodeTypeIndex.size() - 1; }

    // thrust::zip_iterator getIteratorPositionBegin();
    // thrust::zip_iterator getIteratorPositionEnd();
};





#endif