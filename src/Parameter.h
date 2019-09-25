#ifndef PARAMETER_H_
#define PARAMETER_H_


// #include "IParameter.h"


#include <string>


// template<typename T>
class Parameter {
    public:
    Parameter();
    Parameter(std::string _key, T _value);

    std::string getKey() const { return key; };

    // This could later be more flexible to allow for default values.
    double getValue() const { return value; };
    void setValue(T _value);


    private:
    std::string key;
    double value;
    double defaultValue;
};

#endif