#include <fstream>	     
#include <iomanip>
#include <string>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>



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


int main() {

    thrust::host_vector<double> pos_x(4);
    thrust::host_vector<double> pos_y(4);
    thrust::host_vector<double> pos_z(4);

    pos_x[0] = 0.2;
    pos_x[1] = 1.4;
    pos_x[2] = 2.2;
    pos_x[3] = 13.2;

    pos_y[0] = 0.2;
    pos_y[1] = 1.4;
    pos_y[2] = 22.2;
    pos_y[3] = 13.2;
    
    pos_z[0] = 111110.2;
    pos_z[1] = 11.4;
    pos_z[2] = 2.222222222;
    pos_z[3] = 3.141592;

    std::string test = "TestingFileName.vtk";

    std::ofstream ofs;

    ofs.open(test.c_str());    

    thrust::for_each(
        thrust::make_zip_iterator(
            thrust::make_tuple(
                pos_x.begin(),
                pos_y.begin(),
                pos_z.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                pos_x.end(),
                pos_y.end(),
                pos_z.end())),
        functor_output_tuple(ofs));

    ofs.close();

    return 0;
}