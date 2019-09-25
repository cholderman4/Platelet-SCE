#ifndef FUNCTOR_FIX_NODES_H_
#define FUNCTOR_FIX_NODES_H_



// Input:   node ID (of fixed nodes only) 

// Output:  Void 

// Update:  nodeData.force_(x,y,z) set to zero



struct functor_fix_nodes : public thrust::unary_function<unsigned, void> {
    double* vec_force_x;
    double* vec_force_y;
    double* vec_force_z;

    __host__ __device__
        functor_fix_nodes(
            double* _vec_force_x,
            double* _vec_force_y,
            double* _vec_force_z) :

        vec_force_x(_vec_force_x),
        vec_force_y(_vec_force_y),
        vec_force_z(_vec_force_z) {}


    __device__
    void operator()(const unsigned& id) {

        vec_force_x[id] = 0.0;
        vec_force_y[id] = 0.0;
        vec_force_z[id] = 0.0;              
                
    } // End operator()
}; // End struct

#endif