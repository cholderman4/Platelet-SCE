#include "PrintVTK.h"

#include <fstream>	     
#include <iomanip>
#include <thrust/host_vector.h>

#include "mkdir_p.h"
#include "NodeData.h"



PrintVTK::PrintVTK(NodeData& _nodeData) :
    nodeData(_nodeData) {
        setFileName("Test");
}


PrintVTK::PrintVTK(
    NodeData& _nodeData, 
    std::string _fileName) :

    nodeData(_nodeData) 
{
    setFileName(_fileName)
}


void PrintVTK::setFileName(std::string _filename) {
    fileName = _filename;
    setOutputPath();
}

void PrintVTK::setOutputPath() {
    path = generatePath->getOutputPath("AnimationTest/");
    __attribute__ ((unused)) int unused = mkdir_p(path.c_str());
}



void PrintVTK::enrollData(std::shared_ptr<IPrintToFile> d) {
    dataVTK.push_back(d);
}


void PrintVTK::save() {
    // Too big to save every time.
    print();
};


void PrintVTK::print() {

    // Getting the filename for current iteration.
    ++outputCounter;
    
    unsigned digits = ceil(log10(outputCounter + 1));    
    if (digits == 1 || digits == 0) {
        number = "0000" + std::to_string(outputCounter);
    } else if (digits == 2) {
        number = "000" + std::to_string(outputCounter);
    } else if (digits == 3) {
        number = "00" + std::to_string(outputCounter);
    } else if (digits == 4) {
        number = "0" + std::to_string(outputCounter);
    }

    std::string currentFile = path + fileName + number + format;
    
    std::ofstream ofs;

    ofs.open(currentFile.c_str());

    // Print VTK file heading
    ofs << "# vtk DataFile Version 3.0" << std::endl;
    ofs << "Point representing Sub_cellular elem model" << std::endl;
    ofs << "ASCII" << std::endl;
    ofs << "DATASET POLYDATA" << std::endl;

    // Print the various options registered with PrintVTK.
    for (auto d : dataVTK) {
        d->print(ofs);
    }
    
    ofs.close();
}