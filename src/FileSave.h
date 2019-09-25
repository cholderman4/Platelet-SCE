#ifndef FILE_SAVE_H_
#define FILE_SAVE_H_

// Possibly redundant.


#include <string>

#include "ISave.h"

class NodeData;


class FileSave : public ISave {
    protected:
    NodeData& nodeData;
    std::string fileNameDescription;

    public: 
    virtual void save() = 0;
    virtual void print() = 0;

    FileSave(NodeData& _nodeData);
    FileSave(NodeData& _nodeData, std::string _fileNameDescription);

};


#endif // FILE_SAVE_H_