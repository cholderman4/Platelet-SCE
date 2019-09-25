#ifndef PRINT_NODE_POSITIONS_VTK_H_
#define PRINT_NODE_POSITIONS_VTK_H_

#include <fstream>

#include "IPrintToFile.h"

class NodeData;


class PrintNodePositionsVTK : public IPrintToFile {

    private:
    NodeData& nodeData;

    public:
    void print(std::ofstream& ofs);
    
    PrintNodePositionsVTK(NodeData& _nodeData);
};


#endif