#ifndef PLATLET_SYSTEM_H_
#define PLATLET_SYSTEM_H_



class NodeData;
class PlatletSystemController;


class PlatletSystem {

    private:

    NodeData nodeData;

    PlatletSystemController platletSystemController;


    public:

    void runSystem();
};


#endif