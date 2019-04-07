#include "PlatletSystem.h"
// #include "SpringForce.h"


int main() {  

    PlatletSystem platletSystem;

    platletSystem.initializePltSystem(5,5);

    platletSystem.solvePltSystem();
    
    /* // Test values
    unsigned N{3};
    unsigned E{3};
    Node node(N);
    Edge edge(E);

    // Test initialization of nodes and edges.
    node.printPoints();
    
    edge.printConnections();

    Spring_Force(node, edge); */
        
    return 0;
}