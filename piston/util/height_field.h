/*
 * height_field.h
 *
 *  Created on: Oct 21, 2011
 *      Author: ollie
 */

#ifndef HEIGHT_FIELD_H_
#define HEIGHT_FIELD_H_

#include <piston/image3d.h>
#include <piston/util/height_functor.h>

namespace piston {

template <typename IndexType, typename ValueType, typename Space>
struct height_field : public piston::image3d<IndexType, ValueType, Space>
{

    typedef piston::image3d<IndexType, ValueType, Space> Parent;

    typedef thrust::transform_iterator<height_functor,
				       typename Parent::GridCoordinatesIterator> PointDataIterator;
    PointDataIterator point_data_iterator;

    height_field(int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim),
	point_data_iterator(this->grid_coordinates_iterator,
	                    height_functor()){}

    void resize(int xdim, int ydim, int zdim) {
	Parent::resize(xdim, ydim, zdim);
	point_data_iterator = thrust::make_transform_iterator(this->grid_coordinates_iterator,
	                                                      height_functor());
    }

    PointDataIterator point_data_begin() {
	return point_data_iterator;
    }
    PointDataIterator point_data_end() {
	return point_data_iterator + this->NPoints;
    }
};

}
#endif /* HEIGHT_FIELD_H_ */
