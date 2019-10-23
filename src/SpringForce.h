#ifndef SPRING_FORCE_H_
#define SPRING_FORCE_H_


class NodeData;

#include "IFunction.h"
#include "SymmetricForce.h"


class SpringForce : public SymmetricForce, public IFunction {

    private:
    // ==========================================
    // Function specific data (parameters, indexing vectors, etc.)

    // S = # of springs 

    // Vector size is M * N, 
    // where    M = nMaxConnectedSprings
    // and      N = nRegisteredNodes.
    // Each entry corresponds to the ID of 
    // a spring that node is connected to.
    thrust::device_vector<unsigned> springConnectionsByNode;

    // size() = nRegisteredNodes
    // Used to track the number of springs to loop 
    // through for a given node. 
    thrust::device_vector<unsigned> nSpringsConnectedToNode;

    // Used for indexing purposes.
    unsigned nMaxSpringsConnectedToNode{ 20 };

    // Size is 2 * S, 
    // The nth spring is connected to the nodes at
    // nodeConnections[2n] and nodeConnections[2n+1]
    thrust::device_vector<unsigned> nodeConnectionsBySpring;
    

    // Size is S.
    thrust::device_vector<double> len_0;

    // Default value.
    double stiffness{ 30.0 };



    // Do I need this??
    // Equal to nodeConnections.size()/2 ??
    unsigned nSpringEdges{ 0 };
    

    double energy{ 0.0 };
    // ==========================================


    public:

        SpringForce(NodeData& _nodeData);

        void getParameterKeys(ParameterManager& parameterManager);
        void setParameterValues(ParameterManager& parameterManager);


        void execute();
        // void enrollNodeType(std::shared_ptr<INodeType> nodeType);
};




#endif