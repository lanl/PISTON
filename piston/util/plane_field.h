/*
 * plane_field.h
 *
 *  Created on: Oct 24, 2011
 *      Author: ollie
 */

#ifndef PLANE_FIELD_H_
#define PLANE_FIELD_H_

namespace piston {

// TODO: turn this into a factory with different level of caching
template <typename IndexType, typename ValueType, typename Space>
struct sphere_field : public piston::image3d<IndexType, ValueType, Space>
{
    typedef piston::image3d<IndexType, ValueType, Space> Parent;

    typedef typename detail::choose_container<typename Parent::CountingIterator, thrust::tuple<IndexType, IndexType, IndexType> >::type GridCoordinatesContainer;
    GridCoordinatesContainer grid_coordinates_vector;
    typedef typename GridCoordinatesContainer::iterator GridCoordinatesIterator;

    typedef typename detail::choose_container<typename Parent::CountingIterator, ValueType>::type PointDataContainer;
    PointDataContainer point_data_vector;
    typedef typename PointDataContainer::iterator PointDataIterator;

    sphere_field(int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim),
	grid_coordinates_vector(Parent::grid_coordinates_begin(), Parent::grid_coordinates_end()),
	point_data_vector(thrust::make_transform_iterator(grid_coordinates_vector.begin(), plane_functor<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1)),
	                  thrust::make_transform_iterator(grid_coordinates_vector.end(),   plane_functor<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1)))
	                  {}

    void resize(int xdim, int ydim, int zdim) {
	Parent::resize(xdim, ydim, zdim);
	point_data_vector.resize(this->NPoints);
	point_data_vector.assign(thrust::make_transform_iterator(grid_coordinates_vector.begin(), plane_functor<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1)),
	                         thrust::make_transform_iterator(grid_coordinates_vector.end(),   plane_functor<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1)));
    }

    GridCoordinatesIterator grid_coordinates_begin() {
	return grid_coordinates_vector.begin();
    }
    GridCoordinatesIterator grid_coordinates_end() {
	return grid_coordinates_vector.end();
    }

    PointDataIterator point_data_begin() {
	return point_data_vector.begin();
    }
    PointDataIterator point_data_end() {
	return point_data_vector.end();
    }
};

}

#endif /* PLANE_FIELD_H_ */
