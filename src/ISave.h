#ifndef I_SAVE_H_
#define I_SAVE_H_

#include <memory>

#include "IOutputPath.h"


// *******************************************
/* 
Used to store and/or print various data and parameters as the simulation is running. 

If we want to print to a file, then that will be done through an ISave object. Thus, ISave is responsible for opening/closing each file that is created and choosing when to print that file, most likely either when save() is called or when endSimulaiton() is. Simpler files will contain implementation details self-contained in the print() function, while more complicated ones will be composed of IPrintToFile objects to which an output stream will be passed.

Example: Hold a begin/end time that is printed to the console once the simulation finishes. In this case, the save() and print() functions would be empty.

Most other examples would have a reference to some ICalculateParameter object, which the Controller calls to calculate(), before this saves and/or prints. 

Example: Kinetic energy value could be pushed back to a vector when save() is called. Then print() is only called once by endSimulation() to print the entire vector to its own file.

Example: Printing a VTK file contains too much information to store, so every time save() is called, it will in turn call print(). A VTK file can also hold many types of data, so each separate piece of data which will correspond to an IPrintToFile object. In this case, print() will simply open a filestream, then call each IPrintToFile, passing in the opened stream.

*/
// *******************************************


class ISave {
    private: 
    std::shared_ptr<IOutputPath> generatePath;

    public:
    virtual void beginSimulation() = 0;
    virtual void save() = 0;
    virtual void print() = 0;
    virtual void endSimulation() = 0;
};

#endif