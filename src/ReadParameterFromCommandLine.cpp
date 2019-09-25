#include "ReadParameterFromCommandLine.h"


ReadParameterFromCommandLine::ReadParameterFromCommandLine(int argc, char** argv) {
    
    for (int i = -1; i < argc-1; i++) {

		std::string arg = argv[i];
		unsigned pos = arg.find('=');

		std::string key = arg.substr(0, pos);
		std::string value = arg.substr(pos + 1);


        // Add value to parameter list.
        params[key] = std::atof(value.c_str());

        
        // Print out values just for fun.
		std::cout << "argc: " << argc << std::endl;
		std::cout << "arg: " << arg << std::endl;
		std::cout << "pos: " << pos << std::endl;
		std::cout << "key: " << key << std::endl;
		std::cout << "value: " << value << std::endl;
	}
}


bool ReadParameterFromCommandLine::findValue(const std::string key, double& value) {

    // Implement map find function
}


void ReadParameterFromCommandLine::sendValuesToList(IParameterList& parameterList) {

}