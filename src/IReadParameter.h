#ifndef I_READ_PARAMETER_H_
#define I_READ_PARAMETER_H_

#include <string>

class IParameterList;


class IReadParameter {

    public:
    virtual bool findValue(const std::string key, double& value);
    virtual void sendValuesToList(IParameterList& parameterList);

};

#endif // I_RETREIVE_PARAMETER_H_