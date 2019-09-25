#ifndef I_READ_PARAMETER_H_
#define I_READ_PARAMETER_H_

// *******************************************
/* 
Used to read in parameter values, by their key, from some source (XML, command line, itself etc.). Clients can search for a specific key/vlaue, or receive all values into a parameterList. In either case, this class knows how to search/iterate through its source and is responsible to do so (or possibly just return false).

Note that a IParameterList has two options to search through values from this: either it can iterate thorugh its own list and call findValue() each time, or allow this to iterate though its source and setValues()/addValues() for each entry.
*/
// *******************************************


#include <string>

class IParameterList;


class IReadParameter {

    public:
    // Find (and get) a specific value.
    // This may not be supported and implemented with empty function (returning false).
    virtual bool findValue(
                    const std::string key, 
                    double& value) = 0;
    
    // Iterate through source values, calling parameterList.setValue().
    virtual void sendValuesToList(IParameterList& parameterList) = 0;

};

#endif // I_PARAMETER_LIST_H_