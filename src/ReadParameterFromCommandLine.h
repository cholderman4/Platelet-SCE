#ifndef READ_PARAMETER_FROM_COMMAND_LINE_H_
#define READ_PARAMETER_FROM_COMMAND_LINE_H_

#include <map>
#include <string>

#include "IReadParameter.h"

class IParameterList;


class ReadParameterFromCommandLine : public IReadParameter {

    public:
    ReadParameterFromCommandLine(int argc, char** argv);
    
    // IReadParameter functions.
    bool findValue(const std::string key, double& value);
    void sendValuesToList(IParameterList& parameterList);


    private:
    std::map< std::string, double > parameters;
};

#endif // READ_PARAMETER_FROM_COMMAND_LINE_H_