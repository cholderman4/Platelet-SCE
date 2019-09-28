#ifndef PARAMETER_MANAGER_H_
#define PARAMETER_MANAGER_H_

// *******************************************
/* 
A parameter list that manages parameters by holding Parameter objects rather than raw values. This allows us slightly more control over things like default vs. updated values.

We also include the functionality to convert an integer nodeType into a suffix abbreviation. Thus, individual functors only need to know the root of the key and the node types that interact, e.g. "morse_U" is needed for the interaction for node types 0 and 1. This class interprets that information and searches for the key: "morse_U_MI". We will also eventually include checking for just the root key, if the latter, more specific key, is not found.

Note: all values are stored as doubles, so clients (i.e. functor classes) will need to static_cast these where appropriate.
*/
// *******************************************


#include <map>
#include <memory>
#include <string>

#include "IParameterList.h"
#include "Parameter.h"



class ParameterManager : public IParameterList {


    public:
    ParameterManager();
    ~ParameterManager();

    // Add parameters directly. Allows clients more control over default values.
    void addParameter(
        Parameter& parameter); // Not sure whether this should be passed by reference.

    // setValue is overloaded to allow for nodeType abbreviations to be appended to a root key.
    void setValue(
        const std::string key, 
        const double& value, 
        unsigned A, unsigned B);

    bool findValue(
        const std::string key, 
        double& value,
        unsigned A, unsigned B);

    // Called if we want to iterate through all stored parameters and search for values, possibly with some greater degree of control over default parameters, e.g skipping or updating only those values that have already have a default value. (Otherwise, we will just let IReadParameter call sendValuesToList() as it probably has a better way to iterate through its source rather than searching for every key individually.)
    // Possibly used if it is easier to search rather than iterate through a source of parameters.
    // Probably unnecessary.
    void updateParameters(const IReadParameter& readParameters);


    // IParameterList
    void addValue(
        const std::string key, 
        const double& value);

    void setValue(
        const std::string key, 
        const double& value);


    // IReadParameter
    bool findValue(
        const std::string key, 
        double& value);

    void sendValuesToList(IParameterList& parameterList);

    
    

    private:
    std::map<int, std::string> nodeTypeAbbr;

    std::map< std::string, std::shared_ptr<Parameter<double>> > parameters;
};

#endif // PARAMETER_MANAGER_H_