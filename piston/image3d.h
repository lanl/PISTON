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

#ifndef IMAGE3D_H_
#define IMAGE3D_H_

#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace piston
{

// TODO: inherit from image2d?
template <typename MemorySpace =  thrust::detail::default_device_space_tag>
struct image3d
{
    typedef unsigned IndexType;

    IndexType dim0;
    IndexType dim1;
    IndexType dim2;
    IndexType NPoints;
    IndexType NCells;

    // transform from point_id (n) to grid_coordinates (i, j, k)
    struct grid_coordinates_functor : public thrust::unary_function<IndexType, thrust::tuple<IndexType, IndexType, IndexType> >
    {
	IndexType dim0;
	IndexType dim1;
	IndexType dim2;
	IndexType PointsPerLayer;

	grid_coordinates_functor(IndexType dim0, IndexType dim1, IndexType dim2) :
	    dim0(dim0), dim1(dim1), dim2(dim2), PointsPerLayer(dim0*dim1) {}

	__host__ __device__
	thrust::tuple<IndexType, IndexType, IndexType> operator()(const IndexType& point_id) const {
	    const IndexType i = point_id % dim0;
	    const IndexType j = (point_id/dim0) % dim1;
	    const IndexType k = point_id/PointsPerLayer;

	    return thrust::make_tuple(i, j, k);
	}
    };

    typedef typename thrust::counting_iterator<IndexType, MemorySpace> CountingIterator;
    typedef typename thrust::transform_iterator<grid_coordinates_functor, CountingIterator> GridCoordinatesIterator;
    GridCoordinatesIterator grid_coordinates_iterator;

    image3d(IndexType xdim, IndexType ydim, IndexType zdim) :
	dim0(xdim), dim1(ydim), dim2(zdim),
	NPoints(xdim*ydim*zdim),
	NCells((xdim-1)*(ydim-1)*(zdim-1)),
	grid_coordinates_iterator(CountingIterator(0), grid_coordinates_functor(xdim, ydim, zdim)) {}

    GridCoordinatesIterator grid_coordinates_begin() {
	return grid_coordinates_iterator;
    }
    GridCoordinatesIterator grid_coordinates_end() {
	return grid_coordinates_iterator+NPoints;
    }

    typedef GridCoordinatesIterator PhysicalCoordinatesIterator;

    PhysicalCoordinatesIterator physical_coordinates_begin() {
	return grid_coordinates_iterator;
    }
    PhysicalCoordinatesIterator physical_coordinates_end() {
	return grid_coordinates_iterator+this->NPoints;
    }
};

} // namepsace piston

#endif /* IMAGE3D_H_ */
