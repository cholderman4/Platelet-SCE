#ifndef FUNCTOR_KINETIC_ENERGY_H_
#define FUNCTOR_KINETIC_ENERGY_H_



// Input:   velocity 

// Output:  kinetic energy (1/2 * v^2) 

// Update:  None



struct functor_kinetic_energy : public thrust::unary_function<double, double> {
    
    __host__ __device__
    functor_kinetic_energy() {}

    __device__
    double operator()(const double& v) {
        return 0.5 * v * v;
    } 
};

#endif