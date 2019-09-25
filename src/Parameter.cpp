#include "Parameter.h"

Parameter::Parameter() {};

template<typename T>
Parameter<T>::Parameter(std::string _key, T _value) :
    key(_key), value(_value) {};


template<typename T>
void Parameter<T>::setValue(T _value) {
    value = _value;
}