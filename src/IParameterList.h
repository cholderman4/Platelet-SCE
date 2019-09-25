#ifndef I_PARAMETER_LIST_H_
#define I_PARAMETER_LIST_H_

#include <string>

#include "IReadParameter.h"


class IParameterList : public IReadParameter {

    public:
    virtual void addParameter(std::string key, double value);
    virtual void setValue(std::string key, double value);

    virtual bool findValue(const std::string key, double& value);


};

#endif // I_RETREIVE_PARAMETER_H_