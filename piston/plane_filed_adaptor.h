/*
 * plane_filed_adaptor.h
 *
 *  Created on: Nov 15, 2012
 *      Author: ollie
 */

#ifndef PLANE_FILED_ADAPTOR_H_
#define PLANE_FILED_ADAPTOR_H_

#include <piston/image3d.h>
#include <piston/choose_container.h>
#include <piston/util/plane_functor.h>

namespace piston {

template <typename Parent>
struct plane_field_adaptor
{
    Parent& parent;
    typedef unsigned IndexType;

    IndexType dim0;
    IndexType dim1;
    IndexType dim2;
    IndexType NPoints;
    IndexType NCells;

    typedef typename thrust::iterator_traits<typename Parent::GridCoordinatesIterator>::value_type
	    GridCoordinatesType;

    typedef typename Parent::GridCoordinatesIterator GridCoordinatesIterator;
    typedef typename Parent::PhysicalCoordinatesIterator PhysicalCoordinatesIterator;

    typedef typename detail::choose_container<typename Parent::CountingIterator, float>::type PointDataContainer;
    PointDataContainer point_data_vector;
    typedef typename PointDataContainer::iterator PointDataIterator;



    plane_field_adaptor(Parent& parent, float3 origin, float3 normal) :
	parent(parent), dim0(parent.dim0), dim1(parent.dim1), dim2(parent.dim2),
	NPoints(parent.NPoints), NCells(parent.NCells),
	point_data_vector(thrust::make_transform_iterator(parent.physical_coordinates_begin(), plane_functor<GridCoordinatesType, float>(origin, normal)),
	                  thrust::make_transform_iterator(parent.physical_coordinates_end(),   plane_functor<GridCoordinatesType, float>(origin, normal)))
	{}

    PhysicalCoordinatesIterator physical_coordinates_begin() {
	return parent.physical_coordinates_begin();
    }
    PhysicalCoordinatesIterator physical_coordinates_end() {
	return parent.physical_coordinates_end();
    }

    PointDataIterator point_data_begin() {
	return point_data_vector.begin();
    }
    PointDataIterator point_data_end() {
	return point_data_vector.end();
    }
};

}


#endif /* PLANE_FILED_ADAPTOR_H_ */
