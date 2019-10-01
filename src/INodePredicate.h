// *******************************************
/* 
Used to assign the index number of all nodes that meet a certain condition. A client using this interface need only call getNodeID() of the derived class with the proper condition already implemented. 

Note that the nodeID vector is passed by reference, so that the class can make the proper sizing, copying, etc.
*/
// *******************************************


#ifndef I_NODE_PREDICATE_H_
#define I_NODE_PREDICATE_H_

#include <thrust/device_vector.h>


class INodePredicate {
    public:

    virtual void getNodeID(thrust::device_vector<unsigned>& nodeID) = 0;

    virtual ~INodePredicate() {};
    
};

#endif // I_NODE_PREDICATE_H_