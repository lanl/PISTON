/*
 * sphere_field.h
 *
 *  Created on: Oct 21, 2011
 *      Author: ollie
 */

#ifndef SPHERE_FIELD_H_
#define SPHERE_FIELD_H_

#include <piston/image3d.h>
#include <piston/choose_container.h>
#include <piston/util/sphere_functor.h>

namespace piston {

#if 0

template <typename IndexType, typename ValueType>
struct sphere_field : public piston::image3d<IndexType, ValueType, SPACE>
{
    typedef piston::image3d<IndexType, ValueType, SPACE> Parent;

    typedef thrust::transform_iterator<piston::sphere<IndexType, ValueType>,
				       typename Parent::GridCoordinatesIterator> PointDataIterator;
    PointDataIterator point_data_iterator;

    sphere_field(int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim),
	point_data_iterator(this->grid_coordinates_iterator,
	                    sphere<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1)){}
    void resize(int xdim, int ydim, int zdim) {
 	Parent::resize(xdim, ydim, zdim);
 	point_data_iterator = thrust::make_transform_iterator(this->grid_coordinates_iterator,
 	                                                      sphere<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1));
     }

    PointDataIterator point_data_begin() {
	return point_data_iterator;
    }
    PointDataIterator point_data_end() {
	return point_data_iterator+this->NPoints;
    }
};

#else

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
	point_data_vector(thrust::make_transform_iterator(grid_coordinates_vector.begin(), sphere_functor<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1)),
	                  thrust::make_transform_iterator(grid_coordinates_vector.end(),   sphere_functor<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1)))
	                  {}

    void resize(int xdim, int ydim, int zdim) {
	Parent::resize(xdim, ydim, zdim);
	point_data_vector.resize(this->NPoints);
	point_data_vector.assign(thrust::make_transform_iterator(grid_coordinates_vector.begin(), sphere_functor<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1)),
	                         thrust::make_transform_iterator(grid_coordinates_vector.end(),   sphere_functor<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1)));
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

#endif

}

#endif /* SPHERE_FIELD_H_ */
