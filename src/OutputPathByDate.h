#ifndef I_OUTPUT_FILE_NAME_BY_DATE_H_
#define I_OUTPUT_FILE_NAME_BY_DATE_H_

#include <string>

#include "IOutputPath.h"

// *******************************************

// *******************************************


class OutputPathByDate : public IOutputPath {
    public:
    std::string getOutputPath(std::string seed);
    OutputPathByDate();
    reset();

    private:
    std::string date;
    void createDirectory();
    std::string generateOutputPathByDate();
};

#endif // I_OUTPUT_FILE_NAME_BY_DATE_H_