#ifndef PLATLET_SYSTEM_H_
#define PLATLET_SYSTEM_H_



class NodeData;
class PlatletSystemController;


class PlatletSystem {

    public:

    NodeData nodeData;

    PlatletSystemController platletSystemController;

    void runSystem();

    // NodeData& getNodeData();
};


#endif