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

#ifndef IMAGE3D_TO_TETRAHEDRONS_H_
#define IMAGE3D_TO_TETRAHEDRONS_H_

#include <piston/image3d.h>
#include <piston/choose_container.h>

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
	const int dim0;
	const int dim1;
	const int dim2;
	const int points_per_layer;

	int i[8];

	index2index(InputDataSet &input) :
	    dim0(input.dim0), dim1(input.dim1), dim2(input.dim2),
	    points_per_layer(dim0*dim1) {
	    i[0] = 0;
	    i[1] = 1;
	    i[2] = 1 + dim0;
	    i[3] = dim0;

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
		0, 2, 1, 6,
		0, 3, 2, 6,
		0, 7, 3, 6,
		0, 4, 7, 6,
		0, 5, 4, 6,
	    };
	    const int cubeid = pointid/24;
	    return i[vertices_for_tetra[pointid%24]] + cubeid;
	}
    };

    typedef typename thrust::iterator_space<InputPointDataIterator>::type	space_type;
    typedef typename thrust::counting_iterator<int, space_type>			CountingIterator;
    typedef typename thrust::transform_iterator<index2index, CountingIterator>	IndicesIterator;

    typedef thrust::permutation_iterator<InputGridCoordinatesIterator, IndicesIterator> GridCoordinatesIterator;
//    typedef thrust::permutation_iterator<InputPointDataIterator,       IndicesIterator> PointDataIterator;

    int NCells;
    IndicesIterator indices;
    InputDataSet &input;

    GridCoordinatesIterator grid_coordinates_iterator;
//    PointDataIterator	    point_data_iterator;

//    typedef typename detail::choose_container<CountingIterator, float>::type PointDataContainer;
    typedef thrust::device_vector<float> PointDataContainer;
    PointDataContainer point_data_vector;
    typedef typename PointDataContainer::iterator PointDataIterator;

    image3d_to_tetrahedrons(InputDataSet &input) :
	NCells(input.NCells*6),
	input(input),
	indices(CountingIterator(0), index2index(input)),
	grid_coordinates_iterator(input.grid_coordinates_begin(), indices),
//	point_data_iterator(input.point_data_begin(), indices),
	point_data_vector(thrust::make_permutation_iterator(input.point_data_begin(), indices),
	                  thrust::make_permutation_iterator(input.point_data_begin(), indices)+24*input.NCells)
    {}

    GridCoordinatesIterator grid_coordinates_begin() {
	return grid_coordinates_iterator;
    }
    GridCoordinatesIterator grid_coordinates_end() {
	return grid_coordinates_iterator+24*input.NCells;
    }

    PointDataIterator point_data_begin() {
	return point_data_vector.begin();
//	return point_data_iterator;
    }
    PointDataIterator point_data_end() {
	return point_data_vector.end();
//	return point_data_iterator+24*input.NCells;
    }
};

}

#endif /* IMAGE3D_TO_TETRAHEDRONS_H_ */
