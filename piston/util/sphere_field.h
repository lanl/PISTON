/*
Copyright (c) 2011, Los Alamos National Security, LLC
All rights reserved.
Copyright 2011. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL),
which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.

NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.

If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
·         Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
·         Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other
          materials provided with the distribution.
·         Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used
          to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
