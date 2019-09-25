#include "pugixml/include/pugixml.hpp"

#include <string.h>
#include <iostream>

int main()
{
    std::cout << "Loading File \n";
    pugi::xml_document doc;
    if (!doc.load_file("info.xml")) {
        std::cout << "Errorkljdfhjhfd \n";
        return -1;
    } 

    std::cout << "Setting membrane-nodes \n";
    pugi::xml_node root = doc.child("data");
	pugi::xml_node membrane_nodes = root.child("membrane-nodes");

    std::cout << "Starting loop through attributes \n";
    for (pugi::xml_attribute attr = membrane_nodes.first_attribute(); attr; attr = attr.next_attribute())
    {
        std::cout << " " << attr.name() << "=" << attr.value();
    }
    // tag::basic[]

    std::cout << "Starting loop through child nodes.\n";
    for (pugi::xml_node mem_node = membrane_nodes.first_child(); tool; tool = tool.next_sibling())
    {
        std::cout << "Tool:";

        for (pugi::xml_attribute attr = tool.first_attribute(); attr; attr = attr.next_attribute())
        {
            std::cout << " " << attr.name() << "=" << attr.value();
        }

        std::cout << std::endl;
    }
    // end::basic[]
 /*
    std::cout << std::endl;

    // tag::data[]
    for (pugi::xml_node tool = tools.child("Tool"); tool; tool = tool.next_sibling("Tool"))
    {
        std::cout << "Tool " << tool.attribute("Filename").value();
        std::cout << ": AllowRemote " << tool.attribute("AllowRemote").as_bool();
        std::cout << ", Timeout " << tool.attribute("Timeout").as_int();
        std::cout << ", Description '" << tool.child_value("Description") << "'\n";
    }
    // end::data[]

    std::cout << std::endl;

    // tag::contents[]
    std::cout << "Tool for *.dae generation: " << tools.find_child_by_attribute("Tool", "OutputFileMasks", "*.dae").attribute("Filename").value() << "\n";

    for (pugi::xml_node tool = tools.child("Tool"); tool; tool = tool.next_sibling("Tool"))
    {
        std::cout << "Tool " << tool.attribute("Filename").value() << "\n";
    }
    // end::contents[] */
}

// vim:et
