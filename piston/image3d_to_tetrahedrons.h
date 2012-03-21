/*
 * image3d_to_tetrahedrons.h
 *
 *  Created on: Feb 9, 2012
 *      Author: ollie
 */

#ifndef IMAGE3D_TO_TETRAHEDRONS_H_
#define IMAGE3D_TO_TETRAHEDRONS_H_

#include <piston/image3d.h>

namespace piston
{

// TODO: should we put the InputDataSet in template parameter or
// as a parameter to the constructor?
template <typename InputDataSet>
class image3d_to_tetrahedrons
{

public:
    typedef typename InputDataSet::GridCoordinatesIterator InputGridCoordinatesIterator;
    typedef typename InputDataSet::PointDataIterator 	   InputPointDataIterator;

    // construct IndexIterator mapping pointid of tetrahedrons to pointid of image3d
    struct index2index : public thrust::unary_function<int, int>
    {
	// TODO: how often should we update these? Fetch it from input all the time?
	const int xdim;
	const int ydim;
	const int zdim;
	const int points_per_layer;

	int i[8];

	index2index(InputDataSet &input) :
	    xdim(input.xdim), ydim(input.ydim), zdim(input.zdim),
	    points_per_layer(xdim*ydim) {
	    i[0] = 0;
	    i[1] = 1;
	    i[2] = 1 + xdim;
	    i[3] = xdim;

	    i[4] = i[0] + points_per_layer;
	    i[5] = i[1] + points_per_layer;
	    i[6] = i[2] + points_per_layer;
	    i[7] = i[3] + points_per_layer;
	}

	// TODO: this has to be instantiated for every point in the tetrahedrons.
	__host__ __device__
	int operator()(int pointid) {
	    // the main diagonal is from (0,0,0) to (1,1,1), I paid as much attention
	    // as possible to make sure to maintain the orientation of the edges.
	    const int vertices_for_tetra [24] =
	    {
		0, 1, 5, 6,
		0, 1, 2, 6,
		0, 3, 2, 6,
		0, 3, 7, 6,
		0, 4, 7, 6,
		0, 4, 5, 6,
	    };
	    const int cubeid = pointid/24;
	    return i[vertices_for_tetra[pointid%24]] + cubeid;
	}
    };

    typedef typename thrust::iterator_space<InputPointDataIterator>::type	space_type;
    typedef typename thrust::counting_iterator<int, space_type>			CountingIterator;
    typedef typename thrust::transform_iterator<index2index, CountingIterator>	IndicesIterator;

    typedef thrust::permutation_iterator<InputGridCoordinatesIterator, IndicesIterator> GridCoordinatesIterator;
    typedef thrust::permutation_iterator<InputPointDataIterator,       IndicesIterator> PointDataIterator;

    int NCells;
    IndicesIterator indices;
    InputDataSet &input;

    GridCoordinatesIterator grid_coordinates_iterator;
    PointDataIterator	    point_data_iterator;

    image3d_to_tetrahedrons(InputDataSet &input) :
	NCells(input.NCells*6),
	input(input),
	indices(CountingIterator(0), index2index(input)),
	grid_coordinates_iterator(input.grid_coordinates_begin(), indices),
	point_data_iterator(input.point_data_begin(), indices)
    {}

    GridCoordinatesIterator grid_coordinates_begin() {
	return grid_coordinates_iterator;
    }
    GridCoordinatesIterator grid_coordinates_end() {
	return grid_coordinates_iterator+24*input.NCells;
    }

    PointDataIterator point_data_begin() {
	return point_data_iterator;
    }
    PointDataIterator point_data_end() {
	return point_data_iterator+24*input.NCells;
    }
};

}

#endif /* IMAGE3D_TO_TETRAHEDRONS_H_ */
