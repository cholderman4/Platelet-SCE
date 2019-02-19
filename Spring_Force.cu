#include "SystemStructures.h"
#include "PlatletSystem.h"
#include "Spring_Force.h"
#include "functor_spring_force.h"


void Spring_Force(Node& node, SpringEdge& springEdge) {

    unsigned numEdges = springEdge.node_L.size();

    // Temporary vectors to hold forces and node id's before sort/reduce
    thrust::device_vector<unsigned> tempNodeID(numEdges*2, 0);
    thrust::device_vector<double> tempForce_x(numEdges*2, 0.0);
    thrust::device_vector<double> tempForce_y(numEdges*2, 0.0);
    thrust::device_vector<double> tempForce_z(numEdges*2, 0.0);



    // Temporary vector to hold bool directing output to left/right node.
    thrust::device_vector<bool> leftNodeReturn(numEdges*2, 1);
    thrust::fill(leftNodeReturn.begin(), leftNodeReturn.begin()+numEdges, 0);

    

    
    // Create a vector of two copies of the node ID's 
    thrust::device_vector<unsigned> tempNode_L(numEdges*2);
    thrust::device_vector<unsigned> tempNode_R(numEdges*2);

    thrust::copy(springEdge.node_L.begin(), springEdge.node_L.end(), tempNode_L.begin());
    thrust::copy(springEdge.node_L.begin(), springEdge.node_L.end(), tempNode_L.begin()+numEdges);

    thrust::copy(springEdge.node_R.begin(), springEdge.node_R.end(), tempNode_R.begin());
    thrust::copy(springEdge.node_R.begin(), springEdge.node_R.end(), tempNode_R.begin()+numEdges);


    // Print test output.
    std::cout << "Testing temporary vector creation:" << std::endl;
    for(int i=0; i<numEdges*2; ++i) {
        std::cout << "(" << leftNodeReturn[i] << ", " << tempNode_L[i] << ", " << tempNode_R[i] << ")" << std::endl;
    }

    thrust::transform(
        thrust::make_zip_iterator(
            thrust::make_tuple(
                leftNodeReturn.begin(),
                tempNode_R.begin(),
                tempNode_L.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                leftNodeReturn.end(),
                tempNode_R.end(),
                tempNode_L.end())),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                tempNodeID.begin(),
                tempForce_x.begin(),
                tempForce_y.begin(),
                tempForce_z.begin())),
        functor_spring_force(
            thrust::raw_pointer_cast(node.pos_x.data()),
            thrust::raw_pointer_cast(node.pos_y.data()),
            thrust::raw_pointer_cast(node.pos_z.data())));

    std::cout << "Testing temporary force storage:" << std::endl;
    for(int j=0; j<tempNodeID.size(); ++j) {
        std::cout << tempNodeID[j] << ": (" 
            << tempForce_x[j] << ", "
            << tempForce_y[j] << ", " 
            << tempForce_z[j] << ")" << std::endl;
    }

    // Later run thrust::sort_by_key followed by thrust::reduce_by_key.

    thrust::sort_by_key(
        tempNodeID.begin(),
        tempNodeID.end(),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                tempForce_x.begin(),
                tempForce_y.begin(),
                tempForce_z.begin())));

    std::cout << "Testing temporary force sort_by_key." << std::endl;
    for(int j=0; j<tempNodeID.size(); ++j) {
        std::cout << tempNodeID[j] << ": (" 
            << tempForce_x[j] << ", "
            << tempForce_y[j] << ", " 
            << tempForce_z[j] << ")" << std::endl;
    }
    

    // Create vectors to store the output of reduce_by_key.
    thrust::device_vector<unsigned> tempReduceNodeID(node.force_x.size(), 0);
    thrust::device_vector<double> tempReduceForce_x(node.force_x.size(), 0.0);
    thrust::device_vector<double> tempReduceForce_y(node.force_x.size(), 0.0);
    thrust::device_vector<double> tempReduceForce_z(node.force_x.size(), 0.0);

    thrust::equal_to<unsigned> binary_op;
    thrust::reduce_by_key(
        tempNodeID.begin(),
        tempNodeID.end(),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                tempForce_x.begin(),
                tempForce_y.begin(),
                tempForce_z.begin())),
        tempReduceNodeID.begin(),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                tempReduceForce_x.begin(),
                tempReduceForce_y.begin(),
                tempReduceForce_z.begin())),
        binary_op,
        CVec3Add);
                
    /* thrust::transform(
        thrust::make_zip_iterator(
            thrust::make_tuple(
                node.force_x.begin(),
                node.force_y.begin(),
                node.force_z.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                node.force_x.end(),
                node.force_y.end(),
                node.force_z.end())),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                tempReduceForce_x.begin(),
                tempReduceForce_y.begin(),
                tempReduceForce_z.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                node.force_x.begin(),
                node.force_y.begin(),
                node.force_z.begin())),     
        CVec3Add());  */
}