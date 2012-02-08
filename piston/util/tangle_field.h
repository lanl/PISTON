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

#ifndef TANGLE_FIELD_H_
#define TANGLE_FIELD_H_

#include <piston/image3d.h>
#include <piston/choose_container.h>
#include <piston/implicit_function.h>

namespace piston {

// TODO: should we parameterize the ValueType? only float makes sense.
template <typename IndexType, typename ValueType, typename Space>
struct tangle_field : public piston::image3d<IndexType, ValueType, Space>
{
    typedef piston::image3d<IndexType, ValueType, Space> Parent;

    struct tangle_functor : public piston::implicit_function3d<IndexType, ValueType>
    {
	typedef piston::implicit_function3d<IndexType, ValueType> Parent;
	typedef typename Parent::InputType InputType;

        const float xscale;
        const float yscale;
        const float zscale;

        tangle_functor(IndexType xdim, IndexType ydim, IndexType zdim) :
            xscale(2.0f/(xdim - 1.0f)),
            yscale(2.0f/(ydim - 1.0f)),
            zscale(2.0f/(zdim - 1.0f)) {}

        __host__ __device__
        ValueType operator()(InputType pos) const {
            // TODO: move this into GridCoordinates
            // scale and shift such that x, y, z <- [-1,1]
            const float x = 3.0f*(thrust::get<0>(pos)*xscale - 1.0f);
            const float y = 3.0f*(thrust::get<1>(pos)*yscale - 1.0f);
            const float z = 3.0f*(thrust::get<2>(pos)*zscale - 1.0f);

            const float v = (x*x*x*x - 5.0f*x*x +y*y*y*y - 5.0f*y*y +z*z*z*z - 5.0f*z*z + 11.8f) * 0.2f + 0.5f;

            return v;
        }
    };

    typedef typename detail::choose_container<typename Parent::CountingIterator, ValueType>::type PointDataContainer;
    PointDataContainer point_data_vector;
    typedef typename PointDataContainer::iterator PointDataIterator;

    tangle_field(int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim),
	point_data_vector(thrust::make_transform_iterator(Parent::grid_coordinates_begin(), tangle_functor(xdim, ydim, zdim)),
	                  thrust::make_transform_iterator(Parent::grid_coordinates_end(),   tangle_functor(xdim, ydim, zdim)))
	                  {}

    void resize(int xdim, int ydim, int zdim) {
	Parent::resize(xdim, ydim, zdim);

	point_data_vector.resize(this->NPoints);
	point_data_vector.assign(thrust::make_transform_iterator(Parent::grid_coordinates.begin(), tangle_functor(xdim, ydim, zdim)),
	                         thrust::make_transform_iterator(Parent::grid_coordinates.end(),   tangle_functor(zdim, ydim, zdim)));
    }

    PointDataIterator point_data_begin() {
	return point_data_vector.begin();
    }
    PointDataIterator point_data_end() {
	return point_data_vector.end();
    }
};

}

#endif /* TANGLE_FIELD_H_ */
