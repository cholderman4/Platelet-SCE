#ifndef I_CALCULATE_PARAMETER_H_
#define I_CALCULATE_PARAMETER_H_

// *******************************************
/* 
Used to define parameters that need to be calculated as the simulation progresses, but not necessarily saved past the initial time step.

Example: We want to use the kinetic energy to determine if the simulation continues running, or applies the next round of forces. In this case, we don't need to save the energy values, so they are just calculated, checked, then discarded.

Note that if we do need to save() and/or print() some parameter value, then that will be through an ISave object that contains a shared_ptr to this ICalculateParameter object.

Possibly this should be modified to account for parameters that aren't a single value, but correspond to each node, for example.
*/
// *******************************************


class ICalculateParameter {    
    public:
    virtual void calculate() = 0;
    virtual double getParameter() = 0;

    virtual ~ICalculateParameter() {};
};

#endif // I_CALCULATE_PARAMETER_H_