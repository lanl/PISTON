/*
 * image3d.h
 *
 *  Created on: Aug 23, 2011
 *      Author: ollie
 */

#ifndef IMAGE3D_H_
#define IMAGE3D_H_

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

    // TODO: should be in detail namespace with 2d variant.
    struct grid_coordinates_functor : public thrust::unary_function<IndexType, thrust::tuple<IndexType, IndexType, IndexType> >
    {
	const int xdim;
	const int ydim;
	const int zdim;
	const int PointsPerLayer;

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

//    typedef MemorySpace MemorySpace;

    typedef typename thrust::counting_iterator<IndexType, MemorySpace> CountingIterator;
    typedef typename thrust::transform_iterator<grid_coordinates_functor, CountingIterator> GridCoordinatesIterator;

    GridCoordinatesIterator grid_coordinates_iterator;

    image3d(int xdim, int ydim, int zdim) :
	xdim(xdim), ydim(ydim), zdim(zdim),
	NPoints(xdim*ydim*zdim),
	NCells((xdim-1)*(ydim-1)*(zdim-1)),
	grid_coordinates_iterator(CountingIterator(0), grid_coordinates_functor(xdim, ydim, zdim)) {}

    void resize(int xdim, int ydim, int zdim) {
	this->xdim = xdim;
	this->ydim = ydim;
	this->zdim = zdim;
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
