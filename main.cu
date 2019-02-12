#include "PlatletSystem.h"
#include "Spring_Force.h"


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