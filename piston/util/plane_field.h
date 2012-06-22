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

#ifndef PLANE_FIELD_H_
#define PLANE_FIELD_H_

#include <piston/image3d.h>
#include <piston/choose_container.h>
#include <piston/util/plane_functor.h>

namespace piston {

// TODO: turn this into a factory with different level of caching
template <typename MemorySpace>
struct plane_field : public piston::image3d<MemorySpace>
{
    typedef piston::image3d<MemorySpace> Parent;

    //TODO: move this to parent class?
    typedef typename thrust::iterator_traits<typename Parent::GridCoordinatesIterator>::value_type
	    GridCoordinatesType;

    typedef typename Parent::PhysicalCoordinatesIterator PhysicalCoordinatesIterator;

    typedef typename detail::choose_container<typename Parent::CountingIterator, float>::type PointDataContainer;
    PointDataContainer point_data_vector;
    typedef typename PointDataContainer::iterator PointDataIterator;


    plane_field(float3 origin, float3 normal, int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim),
	point_data_vector(thrust::make_transform_iterator(this->physical_coordinates_begin(), plane_functor<GridCoordinatesType, float>(origin, normal)),
	                  thrust::make_transform_iterator(this->physical_coordinates_end(),   plane_functor<GridCoordinatesType, float>(origin, normal)))
	                  {}


    PointDataIterator point_data_begin() {
	return point_data_vector.begin();
    }
    PointDataIterator point_data_end() {
	return point_data_vector.end();
    }
};

}

#endif /* PLANE_FIELD_H_ */
