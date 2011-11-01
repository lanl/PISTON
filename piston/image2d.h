/*
 * image2d.h
 *
 *  Created on: Oct 7, 2011
 *      Author: ollie
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
