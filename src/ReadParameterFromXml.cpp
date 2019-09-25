#include "ReadParameterFromXml.h"

#include <cassert>

#include "pugixml/include/pugixml.hpp"



ReadParameterFromXml::ReadParameterFromXml() {
    fileName = "ParamInfo.xml";
    openfile();
}


ReadParameterFromXml::ReadParameterFromXml(std::string _fileName) :
    fileName(_fileName) {
        openfile();
    };


void ReadParameterFromXml::openFile() {
	parseResult = doc.load_file(fileName.c_str());

    // possibly better to catch this as an exception??
	assert(parseResult); 

    root = doc.child("data");
    params = root.child("parameters");
}


bool ReadParameterFromXml::findValue(const std::string key, double& value) {
    if (auto p = params.child(key)) {
        value = p.text().as_double();
        return true;
    } else {
        return false;
    }
}


void ReadParameterFromXml::sendValuesToList(IParameterList& parameterList) {

    // Iteratate through all parameter values and send to parameterList.
    for (pugi::xml_node p = params.first_child(); p; p = p.next_sibling()) {
        parameterList.setValue(p.name(), p.text().as_double());
    }

}