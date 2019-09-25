#ifndef I_OUTPUT_PATH_H_
#define I_OUTPUT_PATH_H_

#include <string>

// *******************************************
/* 
Interface for determining the output path to be used by classes which print files.

Example: By date: AnimationTest/2019-09-23/14-50-33/

Example: By parameter test: SpringStiffness001
 */
// *******************************************


class IOutputPath {
    public:
    virtual std::string getOutputPath(std::string seed) = 0;
};

#endif // I_OUTPUT_PATH_H_