/*
Copyright (c) 2011, Los Alamos National Security, LLC
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
    	and/or other materials provided with the distribution.
    Neither the name of the Los Alamos National Laboratory nor the names of its contributors may be used to endorse or promote products derived from this
    	software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef IMAGE2D_H_
#define IMAGE2D_H_

#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace piston
{

template <typename IndexType, typename ValueType, typename MemorySpace>
class image2d
{
public:
    int xdim;
    int ydim;
    int NPoints;
    int NCells;

    struct grid_coordinates_functor : public thrust::unary_function<IndexType, thrust::tuple<IndexType, IndexType> >
    {
	int xdim;
	int ydim;

	grid_coordinates_functor(int xdim, int ydim) :
	    xdim(xdim), ydim(ydim) {}

	__host__ __device__
	thrust::tuple<IndexType, IndexType> operator()(IndexType PointId) const {
	    const IndexType x = PointId % xdim;
	    const IndexType y = PointId / xdim;

	    return thrust::make_tuple(x, y);
	}
    };

    typedef typename thrust::counting_iterator<IndexType, MemorySpace> CountingIterator;
    typedef typename thrust::transform_iterator<grid_coordinates_functor, CountingIterator> GridCoordinatesIterator;

    GridCoordinatesIterator grid_coordinates_iterator;

    image2d(int xdim, int ydim) :
	xdim(xdim), ydim(ydim),
	NPoints(xdim*ydim),
	NCells((xdim-1)*(ydim-1)),
	grid_coordinates_iterator(CountingIterator(0), grid_coordinates_functor(xdim, ydim)) {}

    void resize(int xdim, int ydim) {
	this->xdim = xdim;
	this->ydim = ydim;
	this->NPoints = xdim*ydim;
	this->NCells = (xdim-1)*(ydim-1);
	grid_coordinates_iterator = thrust::make_transform_iterator(CountingIterator(0),
	                                                            grid_coordinates_functor(xdim, ydim));
    }

    GridCoordinatesIterator grid_coordinates_begin() {
	return grid_coordinates_iterator;
    }
    GridCoordinatesIterator grid_coordinates_end() {
	return grid_coordinates_iterator+NPoints;
    }
};

} // namepsace piston


#endif /* IMAGE2D_H_ */
