/*
 * threshold.cu
 *
 *  Created on: Sep 21, 2011
 *      Author: ollie
 */

#include <piston/sphere.h>
#include <piston/threshold_geometry.h>

static const int GRID_SIZE = 5;
template <typename IndexType, typename ValueType>
struct height_field : public piston::image3d<IndexType, ValueType, thrust::host_space_tag>
{
    struct height_functor : public piston::implicit_function3d<IndexType, ValueType> {
	typedef piston::implicit_function3d<IndexType, ValueType> Parent;
	typedef typename Parent::InputType InputType;

	__host__ __device__
	ValueType operator()(InputType pos) const {
	    return thrust::get<2>(pos);
	};
    };

    typedef piston::image3d<IndexType, ValueType, thrust::host_space_tag> Parent;

    typedef thrust::transform_iterator<height_functor,
				       typename Parent::GridCoordinatesIterator> PointDataIterator;
    PointDataIterator iter;

    height_field(int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim),
	iter(this->grid_coordinates_iterator,
	     height_functor()){}

    PointDataIterator point_data_begin() {
	return iter;
    }

    PointDataIterator point_data_end() {
	return iter + this->NPoints;
    }
};

template <typename IndexType, typename ValueType>
struct sfield : public piston::image3d<IndexType, ValueType, thrust::host_space_tag>
{
    typedef piston::image3d<IndexType, ValueType, thrust::host_space_tag> Parent;

    typedef thrust::transform_iterator<piston::sphere<IndexType, ValueType>,
				       typename Parent::GridCoordinatesIterator> PointDataIterator;
    PointDataIterator iter;

    sfield(int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim),
	iter(this->grid_coordinates_iterator,
	     piston::sphere<IndexType, ValueType>(0, 0, 0, 1)){}

    PointDataIterator point_data_begin() {
	return iter;
    }

    PointDataIterator point_data_end() {
	return iter+this->NPoints;
    }
};

struct threshold_between : thrust::unary_function<float, bool>
{
    float min_value;
    float max_value;

    threshold_between(float min_value, float max_value) :
	min_value(min_value), max_value(max_value) {}

    __host__ __device__
    bool operator() (float val) const {
	return (min_value <= val) && (val <= max_value);
    }
};

struct print_float4 : public thrust::unary_function<float4, void>
{
	__host__ __device__
	void operator() (float4 p) {
	    std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
	}
};

int main()
{
    sfield<int, float> scalar_field(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    thrust::copy(scalar_field.point_data_begin(), scalar_field.point_data_end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    threshold_geometry<sfield<int, float>, threshold_between> threshold(scalar_field, threshold_between(0, 1));
    threshold();

    thrust::for_each(threshold.verticesBegin(), threshold.verticesEnd(), print_float4());

    return 0;
}
