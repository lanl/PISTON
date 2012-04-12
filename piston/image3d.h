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

template <typename IndexType, typename ValueType, typename MemorySpace>
class image3d
{
public:
    int xdim;
    int ydim;
    int zdim;
    int NPoints;
    int NCells;

    struct grid_coordinates_functor : public thrust::unary_function<IndexType, thrust::tuple<IndexType, IndexType, IndexType> >
    {
	int xdim;
	int ydim;
	int zdim;
	int PointsPerLayer;

	grid_coordinates_functor(int xdim, int ydim, int zdim) :
	    xdim(xdim), ydim(ydim), zdim(zdim), PointsPerLayer(xdim*ydim) {}

	__host__ __device__
	thrust::tuple<IndexType, IndexType, IndexType> operator()(IndexType PointId) const {
	    const IndexType x = PointId % xdim;
	    const IndexType y = (PointId/xdim) % ydim;
	    const IndexType z = PointId/PointsPerLayer;

	    return thrust::make_tuple(x, y, z);
	}
    };

    typedef typename thrust::tuple<IndexType, IndexType, IndexType> GridCoordinatesType;
    typedef typename thrust::counting_iterator<IndexType, MemorySpace> CountingIterator;
    typedef typename thrust::transform_iterator<grid_coordinates_functor, CountingIterator> GridCoordinatesIterator;

    GridCoordinatesIterator grid_coordinates_iterator;

    image3d(int xdim, int ydim, int zdim) :
	xdim(xdim), ydim(ydim), zdim(zdim),
	NPoints(xdim*ydim*zdim),
	NCells((xdim-1)*(ydim-1)*(zdim-1)),
	grid_coordinates_iterator(CountingIterator(0), grid_coordinates_functor(xdim, ydim, zdim)) { }

    void resize(int xdim, int ydim, int zdim) {
	this->xdim = xdim;
	this->ydim = ydim;
	this->zdim = zdim;
	this->NPoints = xdim*ydim*zdim;
	this->NCells  = (xdim-1)*(ydim-1)*(zdim-1);
	grid_coordinates_iterator = thrust::make_transform_iterator(CountingIterator(0),
	                                                            grid_coordinates_functor(xdim, ydim, zdim));
    }

    GridCoordinatesIterator grid_coordinates_begin() {
	return grid_coordinates_iterator;
    }
    GridCoordinatesIterator grid_coordinates_end() {
	return grid_coordinates_iterator+NPoints;
    }
};

} // namepsace piston

#endif /* IMAGE3D_H_ */
