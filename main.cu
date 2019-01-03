#include "calculateForces.h"
#include "thrust/device_vector.h"
#include "NodeInfo.h"
#include <ostream>

class Node;
class Edge;

 //Testing commit
void setSpringForce(Node& node, Edge& edge) {

    unsigned numEdges = edge.node_L.size();



    // Temporary vectors to hold forces and node id's before sort/reduce
    thrust::device_vector<unsigned> tempNodeID(numEdges*2, 47);
    thrust::device_vector<double> tempForce_x(numEdges*2, 4.7f);
    thrust::device_vector<double> tempForce_y(numEdges*2, 4.7f);
    thrust::device_vector<double> tempForce_z(numEdges*2, 4.7f);



    // Temporary vector to hold bool directing output to left/right node.
    thrust::device_vector<bool> leftNodeReturn(numEdges*2, 1);
    thrust::fill(leftNodeReturn.begin(), leftNodeReturn.begin()+numEdges, 0);

    

    
    // Create a vector of two copies of the node ID's 
    thrust::device_vector<unsigned> node_L(numEdges*2);
    thrust::device_vector<unsigned> node_R(numEdges*2);

    thrust::copy(edge.node_L.begin(), edge.node_L.end(), node_L.begin());
    thrust::copy(edge.node_L.begin(), edge.node_L.end(), node_L.begin()+numEdges);

    thrust::copy(edge.node_R.begin(), edge.node_R.end(), node_R.begin());
    thrust::copy(edge.node_R.begin(), edge.node_R.end(), node_R.begin()+numEdges);


    // Print test output.
    std::cout << "Testing temporary vector creation:" << std::endl;
    for(int i=0; i<numEdges*2; ++i) {
        std::cout << "(" << leftNodeReturn[i] << ", " << node_L[i] << ", " << node_R[i] << ")" << std::endl;
    }

    thrust::transform(
        thrust::make_zip_iterator(
            thrust::make_tuple(
                leftNodeReturn.begin(),
                node_R.begin(),
                node_L.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                leftNodeReturn.end(),
                node_R.end(),
                node_L.end())),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                tempNodeID.begin(),
                tempForce_x.begin(),
                tempForce_y.begin(),
                tempForce_z.begin())),
            getForcesFunctor(
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

}

int main() {  
    
    // Test values
    unsigned N{3};
    unsigned E{3};
    Node node(N);
    Edge edge(E);

    // Test initialization of nodes and edges.
    node.printPoints();
    
    edge.printConnections();

    setSpringForce(node, edge);
        
    return 0;
}