#ifndef I_PRINT_TO_FILE_H_
#define I_PRINT_TO_FILE_H_

#include <fstream>

// *******************************************
/* Used to print some piece of data to an already existing (and possibly more complicated) file. This file is managed by an ISave object, which itself contains pointers to multiple IPrintToFile objects. These files can then be modified by adding or deleting IPrintToFile objects. 

Example: Printing a VTK file through the PrintToVTK : ISave object. PrintToVTK will hold pointers to various IPrintToFile objects, each of which holds implementation to output their own piece of data (positions, forces, node types, etc.). When print() is called, PrintToVTK simply opens a file, then passes the output stream to each IPrintToFile call before closing the stream.

Example: we want a file that is a dump of all parameter data. Each parameter (or group of parameters) will consist of its own IPrintToFile object (themselves pointing to an ICalculateParameter object)that is composed within and called from an ISave object.

However, if we want a file that just displays e.g. kinetic energy, then that will be output to its own file under the print() command without the need to invoke IPrintToFile (since it would just be a single piece of data).
*/
// *******************************************


class IPrintToFile {
    public:
    virtual void print(std::ofstream& ofs) = 0;
};

#endif