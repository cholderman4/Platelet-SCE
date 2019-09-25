#ifndef FUNCTOR_OUTPUT_FILE_H_
#define FUNCTOR_OUTPUT_FILE_H_

#include <fstream>	     
#include <iomanip>
#include <string>


// Input:   (x,y,z) tuple

// Output:   

// Update:  

typedef thrust::tuple<double, double, double> CVec3;


struct functor_output_tuple : public thrust::unary_function<CVec3, void> {

    std::ofstream& ofs;

    __host__ 
        functor_output_tuple(std::ofstream& _ofs) : ofs(_ofs) {}

    __host__
    void operator()(const CVec3& vec) {

        double vec_x = thrust::get<0>(vec);
        double vec_y = thrust::get<1>(vec);
        double vec_z = thrust::get<2>(vec);

        ofs << std::setprecision(5) << std::fixed << 
            vec_x << " " << 
            vec_y << " " << 
            vec_z << " " << '\n' << std::fixed;                
    } 
}; 

template<typename T>
struct functor_output_value : public thrust::unary_function<T, void> {

    std::ofstream& ofs;

    __host__ 
        functor_output_value(std::ofstream& _ofs) : ofs(_ofs) {}

    __host__
    void operator()(const T& val) {
        ofs << std::setprecision(5) << std::fixed << 
            val << '\n' << std::fixed;                
    } 
}; 

#endif