/*
 * image3d_to_tet.cu
 *
 *  Created on: Mar 12, 2012
 *      Author: ollie
 */

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include "piston/util/height_field.h"
#include "piston/image3d_to_tetrahedrons.h"
#include "piston/marching_tetrahedron.h"

using namespace piston;

//#define SPACE thrust::host_space_tag
#define SPACE thrust::detail::default_device_space_tag

template <typename Space>
struct pointid_field : public image3d<int, int, Space>
{
    typedef piston::image3d<int, int, Space> Parent;

    typedef thrust::counting_iterator<int, Space> PointDataIterator;
    PointDataIterator point_data_iterator;

    pointid_field(int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim), point_data_iterator(0)  {}

    PointDataIterator point_data_begin() {
	return point_data_iterator;
    }
    PointDataIterator point_data_end() {
	return point_data_iterator+this->NPoints;
    }

};


void print_tuple3(thrust::tuple<int, int, int> t)
{
    std::cout << "(" << thrust::get<0>(t) << ", " << thrust::get<1>(t) << ", " << thrust::get<2>(t) << ")" << std::endl;
}

struct print_float4
{
    void operator()(float4 v) {
        std::cout << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
    }
};


int main()
{
//    pointid_field<SPACE> field(3,2,2);
    height_field<int, float, SPACE> field(2,2,2);

//    thrust::copy(field.point_data_begin(), field.point_data_end(),
//                 std::ostream_iterator<int>(std::cout, " "));
//    std::cout << std::endl;

    image3d_to_tetrahedrons<height_field<int, float, SPACE> > tetra(field);

    thrust::host_vector<int> vec(tetra.point_data_begin(), tetra.point_data_end());
    std::cout << "vec.size(): " << vec.size() << std::endl;

    for (int i = 0; i < vec.size(); i++) {
	if ((i % 4) == 0)
	    std::cout << std::endl;
	std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

    thrust::host_vector<thrust::tuple<int, int, int> > coordinates(tetra.grid_coordinates_begin(),
                                                                   tetra.grid_coordinates_end());
    thrust::for_each(coordinates.begin(), coordinates.end(), print_tuple3);

    marching_tetrahedron<image3d_to_tetrahedrons<height_field<int, float, SPACE> >,
			 image3d_to_tetrahedrons<height_field<int, float, SPACE> > > isosurface(tetra, tetra, 0.5f);
    isosurface();

    thrust::host_vector<float4> vertices(isosurface.vertices_begin(),
                                         isosurface.vertices_end());
    thrust::for_each(vertices.begin(), vertices.end(), print_float4());
}
