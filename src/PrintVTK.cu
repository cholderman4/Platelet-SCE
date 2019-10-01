#include "PrintVTK.h"

#include <fstream>	     
#include <iomanip>

#include "IOutputPath.h"
#include "mkdir_p.h"
#include "NodeData.h"



PrintVTK::PrintVTK(
    NodeData& _nodeData, 
    std::shared_ptr<IOutputPath> outputPathGenerator,
    std::string _seedPath = "AnimationTest/",
    std::string _seedFileName = "Test") :

    nodeData(_nodeData),
    seedPath(_seedPath),
    seedFileName(_seedFileName) {
        // Allow generator to set the output path.
        path = outputPathGenerator->getOutputPath(seedPath);

        // Make sure that the directory of path actually exists.
        // This could possibly be moved to OutputPath itself.
        __attribute__ ((unused)) int unused = mkdir_p(path.c_str());
}


void PrintVTK::enrollData(std::shared_ptr<IPrintToFile> d) {
    dataVTK.push_back(d);
}


void PrintVTK::beginSimulation() {
    print();
}
 

void PrintVTK::endSimulation() {
    print();
}


void PrintVTK::save() {
    // Too big to save every time.
    print();
};


void PrintVTK::print() {

    // Getting the filename for current iteration.
    ++outputCounter;
    
    unsigned digits = ceil(log10(outputCounter + 1));
    std::string number;  
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
    // Each data option is responsible to print its own heading.
    for (auto d : dataVTK) {
        d->print(ofs);
    }
    
    ofs.close();
}