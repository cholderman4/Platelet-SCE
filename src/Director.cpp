#include "Director.h"

#include "pugixml/include/pugixml.hpp"



Director(
    std::unique_ptr<PlatletSystemBuilder> _pltBuilder,
    std::string _nodeDataFile, 
    std::string _paramDataFile) :
    platletSystemBuilder(std::move(_pltBuilder)),
    nodeDataFile(_nodeDataFile),
    paramDataFile(_paramDataFile) {
        
}



Director(
    std::unique_ptr<PlatletSystemBuilder> _pltBuilder,
    std::string _nodeDataFile, 
    std::string _paramDataFile,
    int argc, char** argv) :
    platletSystemBuilder(std::move(_pltBuilder)),
    nodeDataFile(_nodeDataFile),
    paramDataFile(_paramDataFile) {


}


void Director::createPlatletSystem() {
    setNodeData();

    initializeFunctors();

    setParameterInputs();

    setParameterOutputs();
    
    setSaveStates();

}


void Director::setNodeData() {

    pugi::xml_document doc;
	pugi::xml_parse_result parseResult = doc.load_file(nodeDataFile.c_str());

    assert(parseResult); 

    pugi::xml_node root = doc.child("data");
	pugi::xml_node memNodes = root.child("membrane-nodes");
    pugi::xml_node intNodes = root.child("interior-nodes");
	pugi::xml_node links = root.child("links");

    // Check that no crucial data is missing.
	// assert(root && memNodes && links);


    // *****************************************************

    // std::cout << "Adding membrane nodes\n";
    // Add membrane nodes.

    // Maybe add count as an attribute value.

    double x, y, z; //variables to be used reading in data.

	for (auto memNode = memNodes.child("mem-node"); memNode; memNode = memNode.next_sibling("mem-node")) {
		const char* text = memNode.text().as_string();

		assert( (3 == sscanf(text, "%lf %lf %lf", &x, &y, &z)) );

		__attribute__ ((unused)) int unused = platletSystemBuilder->addMembraneNode( glm::dvec3(x, y, z) );
    }


    // std::cout << "Adding interior nodes\n";
    // Add interior nodes.

    // Maybe add count as an attribute value.
    for (auto intNode = intNodes.child("int-node"); intNode; intNode = intNode.next_sibling("int-node")) {
		const char* text = intNode.text().as_string();

		assert( (3 == sscanf(text, "%lf %lf %lf", &x, &y, &z)) );

		__attribute__ ((unused)) int unused = platletSystemBuilder->addInteriorNode( glm::dvec3(x, y, z) );
    }


    // *****************************************************

    // std::cout << "Adding membrane spring connections\n";
    // Add membrane spring connections.

    unsigned n1, n2; //variables to be used reading in data.

	for (auto link = links.child("link"); link; link = link.next_sibling("link")) {
		const char* text = link.text().as_string();

		assert( (2 == sscanf(text, "%u %u", &n1, &n2)) );

		__attribute__ ((unused)) int unused = platletSystemBuilder->addMembraneEdge( n1, n2 );
    }


    // This is optional. It's probably easier to fix the nodes in c++ than Matlab.
    // std::cout << "Setting fixed nodes\n";
    // Read in fixed nodes.
    /* pugi::xml_node fixedNodes = root.child("fixed");
	if (fixedNodes) {
		for (auto node = fixedRoot.child("nodeID"); node; node = node.next_sibling("nodeID"))
			platletSystemBuilder->fixNode(node.text().as_uint());
	} */

}


void Director::initializeFunctors() {
    platletSystemBuilder->initializeFunctors();
}


void Director::setInputs() {
    platletSystemBuilder->setInputs();
}


void Director::setOutputs() {
    platletSystemBuilder->setOutputs();
}


std::unique_ptr<PlatletSystem> getPlatletSystem() {
    return platletSystemBuilder->getSystem().release();
}
