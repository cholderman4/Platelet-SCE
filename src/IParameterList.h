#ifndef I_PARAMETER_LIST_H_
#define I_PARAMETER_LIST_H_


// *******************************************
/* 
A specific case of an IReadParameter object (hence the inheritance) that internally manages the source of parameter key/values (e.g. a map of some kind). This means we can search/iterate through parameters as before, but additionally we can add, set, or otherwise change those values.
*/
// *******************************************


#include <string>

#include "IReadParameter.h"


class IParameterList : public IReadParameter {

    public:

    // Possibly duplicate functionality.
    // Maybe clients want different behavior if the value already exists or not.
    // Possibly one or both of these could be changed to return bool.
    virtual void addValue(
        const std::string key, 
        const double& value) = 0;

    virtual void setValue(
        const std::string key, 
        const double& value) = 0;


    // IReadParameter
    virtual bool findValue(
        const std::string key, 
        double& value) = 0;

    virtual void sendValuesToList(IParameterList& parameterList) = 0;

    virtual ~IParameterList() {};
};

#endif // I_PARAMETER_LIST_H_