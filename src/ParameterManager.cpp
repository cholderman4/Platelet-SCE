#include "ParameterManager.h"

#include <memory>

#include "IReadParameter.h"


ParameterManager::ParameterManager() {
    nodeTypeAbbr[0] = "M";
    nodeTypeAbbr[1] = "I";
};


bool ParameterManager::findValue(const std::string key, double& value) {
    // Include error checking!!!
    value = params[key]->getValue();
    return true;
}


void ParameterManager::addParameter(const std::string key, const double value) {
    params[key] = std::make_shared<Parameter<double>>(key, value);
}


void ParameterManager::setValue(const std::string key, const double value) {
    params[key]->setValue(value);
}


void ParameterManager::updateParameters(const IReadParameter& paramList) {
    for (auto it = parameters.begin(); it != parameters.end(); ++it) {
        double value;
        if (findValue(it->first, value)) {
            it->second->setValue(value);
        } else {
            // Value not found.
            // Resort to default value.
            // May or not be OK.
            std::cout << "Warning: "
                << it->first << " not found. Using default value: "
                << it->second->getValue() << '\n';
        }
    }
}