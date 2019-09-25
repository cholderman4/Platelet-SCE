#ifndef READ_PARAMETER_FROM_XML_H
#define READ_PARAMETER_FROM_XML_H

#include <string>

#include "IReadParameter.h"

class IParameterList;



class ReadParameterFromXml : public IReadParameter {

    public:
    ReadParameterFromXml();
    ReadParameterFromXml(std::string _fileName);
    bool findValue(const std::string key, double& value);
    void sendValuesToList(IParameterList& parameterList);

    private:
    std::string fileName;
    pugi::xml_document doc;
    pugi::xml_parse_result parseResult;
    pugi::xml_node root;
    pugi::xml_node params;

    void openFile();
};

#endif // I_RETREAD_PARAMETER_FROM_XML_HREIVE_PARAMETER_H_