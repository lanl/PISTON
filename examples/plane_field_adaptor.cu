/*
 * plane_field_adaptor.cu
 *
 *  Created on: Nov 15, 2012
 *      Author: ollie
 */
#include <piston/piston_math.h>
#include <piston/util/tangle_field.h>
#include <piston/plane_filed_adaptor.h>

#define SPACE thrust::host_space_tag
using namespace piston;

struct print_tuple
{
    template <typename Tuple>
    __host__ __device__
    void operator ()(Tuple xyz) {
	std::cout << "("
		  << thrust::get<0>(xyz) << ", "
		  << thrust::get<1>(xyz) << ", "
		  << thrust::get<2>(xyz) << ")" << std::endl;
    }
};

struct print_float
{
    __host__ __device__
    void operator()(float x) {
	std::cout << x << " ";
    }
};
int
main()
{
    tangle_field<SPACE> tangle(4,4,4);

    plane_field_adaptor<tangle_field<SPACE> > plane(tangle,
                                                    make_float3(0, 0, 0),
                                                    make_float3(0, 0, 1));

    thrust::for_each(tangle.physical_coordinates_begin(),
                     tangle.physical_coordinates_end(), print_tuple());

    thrust::for_each(plane.physical_coordinates_begin(),
                     plane.physical_coordinates_end(), print_tuple());

    thrust::copy(tangle.point_data_begin(),
                 tangle.point_data_end(),
                 std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    thrust::copy(plane.point_data_begin(),
                 plane.point_data_end(),
                 std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}



