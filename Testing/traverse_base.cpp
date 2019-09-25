#include "pugixml/include/pugixml.hpp"

#include <string.h>
#include <iostream>

int main()
{
    pugi::xml_document doc;
    if (!doc.load_file("paramInfo.xml")) return -1;

    pugi::xml_node tools = doc.child("data").child("parameters");

    // tag::basic[]
    for (pugi::xml_node tool = tools.first_child(); tool; tool = tool.next_sibling())
    {
        std::cout << "Tool:" << tool.name();
        std::cout << tool.text().as_double() << '\n';


        for (pugi::xml_attribute attr = tool.first_attribute(); attr; attr = attr.next_attribute())
        {
            std::cout << " " << attr.name() << "=" << attr.value();
        }

        std::cout << std::endl;
    }
    // end::basic[]

    std::cout << std::endl;

    

    
}

// vim:et
