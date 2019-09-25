#ifndef PRINT_NODES_VTK_H_
#define PRINT_NODES_VTK_H_


#include <memory>
#include <string>
#include <vector>

#include "IPrintToFile.h"
#include "ISave.h"


class NodeData;


class PrintVTK : public ISave {

    private:
    // By default, this will either print all the nodes, or one type of node.
    // To print select types, instantiate a new object for each type.
    NodeData& nodeData;

    // Holds options to print.
    std::vector< std::shared_ptr<IPrintToFile> > dataVTK;    

    // Variables needed for filename info.
    unsigned outputCounter{ 0 };
    std::string fileName;
    std::string format = ".vtk";
    std::string number;
    std::string path = "AnimationTest/";


    public:
    PrintVTK(NodeData& _nodeData);
    PrintVTK(NodeData& _nodeData, std::string _fileName);
    
    // ISave
    void beginSimulation();
    void save();
    void print();
    void endSimulation();

    // PrintVTK 
    void setFileName(std::string _filename);
    void setOutputPath(std::string _outputPath);
    void enrollData(std::shared_ptr<IPrintToFile> d);
};


#endif