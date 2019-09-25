#ifndef PARAMETER_MANAGER_H_
#define PARAMETER_MANAGER_H_

#include <map>
#include <memory>
#include <string>

#include "IParameterList.h"
#include "Parameter.h"



class ParameterManager : public IParameterList {


    public:
    ParameterManager();

    // IReadParameter
    bool findValue(const std::string key, double& value);

    // IParameterList
    void addParameter(const std::string key, const double value);
    void setValue(const std::string key, const double value);

    // setValue is overloaded to allow for nodeType abbreviations to be appended to a base key.
    void setValue(const std::string key, const double value, unsigned A, unsigned B);

    void updateParameters(const IReadParameter& paramList);

    

    private:
    std::map<int, std::string> nodeTypeAbbr;

    std::map< std::string, std::shared_ptr<Parameter<double>> > parameters;
};

#endif