#ifndef PRINT_NODE_TYPE_VTK_H_
#define PRINT_NODE_TYPE_VTK_H_

#include <fstream>

#include "IPrintToFile.h"

class NodeData;


class PrintNodeTypeVTK : public IPrintToFile {

    private:
    NodeData& nodeData;

    public:
    void print(std::ofstream& ofs);
    
    PrintNodeTypeVTK(NodeData& _nodeData);
};


#endif