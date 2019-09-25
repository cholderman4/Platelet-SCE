#ifndef NODE_TYPE_H_
#define NODE_TYPE_H_

class NodeData;

#include <memory>


class NodeType {

protected:
// Needs access 
    std::shared_ptr<NodeData> nodeData;
    unsigned indexBegin;
    unsigned indexEnd;
    unsigned myTypeID;

public:

    unsigned getIndexBegin() const { return indexBegin; }
    unsigned getIndexEnd() const {return indexEnd; }
    
    void updateIndex();
};

#endif