#ifndef DIRECTOR_H_
#define DIRECTOR_H_

#include <memory>
#include <string>
#include <vector>

class IReadParameter;
class PlatletSystemBuilder;


class Director {

    public: 

    // Constructors
    Director(
        std::unique_ptr<PlatletSystemBuilder> _pltBuilder,
        std::string _nodeDataFile, 
        std::string _paramDataFile);
    Director(
        std::unique_ptr<PlatletSystemBuilder> _pltBuilder,
        std::string _nodeDataFile, 
        std::string _paramDataFile,
        int argc, char** argv);

    void createPlatletSystem();
    std::unique_ptr<PlatletSystem> getPlatletSystem();
    

    private:
    std::string nodeDataFile;
    std::string paramDataFile;

    std::vector<std::shared_ptr<IReadParameter>> readParams;

    std::unique_ptr<PlatletSystemBuilder> platletSystemBuilder;

    void setNodeData();
    void initializeFunctors();
    void setInputs();
    void setOutputs();


};




#endif // DIRECTOR_H_