/*
 * extraction.cu
 *
 *  Created on: Aug 26, 2011
 *      Author: ollie
 */

#include <iostream>

#include <thrust/host_vector.h>

#include <piston/image3d.h>

static const int GRID_SIZE = 3;
static const int NPoints = GRID_SIZE*GRID_SIZE*GRID_SIZE;

template <typename IndexType>
struct inside_sphere : public thrust::unary_function<thrust::tuple<IndexType, IndexType, IndexType>, bool>
{
    typedef thrust::tuple<IndexType, IndexType, IndexType> InputType;

    const IndexType x_o;
    const IndexType y_o;
    const IndexType z_o;
    const IndexType radius;

    inside_sphere(IndexType x, IndexType y, IndexType z, IndexType radius) :
	x_o(x), y_o(y), z_o(z), radius(radius) {}

    __host__ __device__
    bool operator()(InputType pos) const {
	const IndexType x = thrust::get<0>(pos);
	const IndexType y = thrust::get<1>(pos);
	const IndexType z = thrust::get<2>(pos);
	const IndexType xx = x - x_o;
	const IndexType yy = y - y_o;
	const IndexType zz = z - z_o;
	return (xx*xx + yy*yy + zz*zz < radius*radius);
    }
};

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
//    typedef typename Parent::MemorySpace MemorySpace;

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

#if 1
// TODO: there should be another abstract class that declares the interface of both image3d and extraction,
// there are both some kind of 3D point data source which provide grid coordinates and point data iterators.
template <typename InputDataSet, typename Predicate>
struct extraction
{
    // valid_cell_enum
    // valid_cell_indices
    typedef thrust::host_vector<int> IndicesContainer;
    typedef typename IndicesContainer::iterator IndicesIterator;
    // GridCoordinatesIterator should be either a permutation iterator of valid_cell_indices and InputData::GridCoordinatesIterator
    // or simply a copy of InputData::GridCoordinates
    typedef typename thrust::permutation_iterator<typename InputDataSet::GridCoordinatesIterator, IndicesIterator> GridCoordinatesIterator;

    extraction(InputDataSet input, Predicate predicate) {

    }


};
#endif

int main()
{
    height_field<int, int> height(GRID_SIZE, GRID_SIZE, GRID_SIZE);

    thrust::host_vector<thrust::tuple<int, int, int> > coord_out(NPoints);
    thrust::host_vector<int> scalar_out(NPoints);
    typedef typename thrust::zip_iterator<thrust::tuple<thrust::host_vector<thrust::tuple<int, int, int> >::iterator,
					                thrust::host_vector<int>::iterator>  > OutputIterator;
    OutputIterator out_end =
	    thrust::copy_if(thrust::make_zip_iterator(thrust::make_tuple(height.grid_coordinates_begin(), height.point_data_begin())),
	                    thrust::make_zip_iterator(thrust::make_tuple(height.grid_coordinates_end(),   height.point_data_end())),
	                    height.grid_coordinates_begin(),
	                    thrust::make_zip_iterator(thrust::make_tuple(coord_out.begin(), scalar_out.begin())),
	                    inside_sphere<float>(0, 0, 0, 2));
    std::cout << "npoints: " << thrust::distance(thrust::make_zip_iterator(thrust::make_tuple(coord_out.begin(), scalar_out.begin())), out_end) << std::endl;
    return 0;
}
