#ifndef SYMMETRIC_FORCE_H_
#define SYMMETRIC_FORCE_H_

#include <vector>

#include "NodeTypeOperation.h"


class SymmetricForce : public NodeTypeOperation {

    private:

    std::vector<unsigned> nodeInteractionA;
    std::vector<unsigned> nodeInteractionB;


    public:
    SymmetricForce(NodeData& _nodeData);
    ~SymmetricForce();

    void registerNodeInteraction(unsigned nodeTypeA, unsigned nodeTypeB);

};

#endif // SYMMETRIC_FORCE_H_