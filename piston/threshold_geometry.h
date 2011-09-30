/*
 * threshold_geometry.h
 *
 *  Created on: Sep 21, 2011
 *      Author: ollie
 */

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/tuple.h>

#include <piston/image3d.h>
#include <piston/sphere.h>
#include <piston/cutil_math.h>
#include <piston/choose_container.h>

using namespace piston::detail;

template <typename InputDataSet, typename ThresholdFunction>
struct threshold_geometry
{
    typedef typename InputDataSet::PointDataIterator InputPointDataIterator;

    typedef typename thrust::iterator_difference<InputPointDataIterator>::type	diff_type;
    typedef typename thrust::iterator_space<InputPointDataIterator>::type	space_type;
    typedef typename thrust::iterator_value<InputPointDataIterator>::type	value_type;

    typedef typename thrust::counting_iterator<int, space_type>	CountingIterator;

    typedef typename choose_container<InputPointDataIterator, int>::type  IndicesContainer;
    typedef typename choose_container<InputPointDataIterator, bool>::type ValidFlagsContainer;
    typedef typename choose_container<InputPointDataIterator, float4>::type VerticesContainer;

    typedef typename IndicesContainer::iterator IndicesIterator;
    typedef typename ValidFlagsContainer::iterator ValidFlagsIterator;
    typedef typename VerticesContainer::iterator VerticesIterator;

    InputDataSet &input;
    ThresholdFunction threshold;

    ValidFlagsContainer valid_cell_flags;
    IndicesContainer    valid_cell_enum;
    IndicesContainer    valid_cell_indices;
    IndicesContainer    num_boundary_cell_neighbors;
    ValidFlagsContainer boundary_cell_flags;
    IndicesContainer    boundary_cell_enum;
    IndicesContainer    boundary_cell_indices;
    VerticesContainer   vertices;

    threshold_geometry(InputDataSet &input, ThresholdFunction threshold) :
	input(input), threshold(threshold) {}

    void operator()() {
	const int NCells = input.NCells;

	std::cout << std::endl;

	valid_cell_flags.resize(NCells);

	// test and enumerate cells that pass threshold
	thrust::transform(CountingIterator(0), CountingIterator(0)+NCells,
	                  valid_cell_flags.begin(),
	                  threshold_cell(input, threshold));
	thrust::host_vector<bool> dummy(valid_cell_flags.begin(), valid_cell_flags.end());
	thrust::copy(dummy.begin(), dummy.end(), std::ostream_iterator<bool>(std::cout, " "));
	std::cout << std::endl;

	valid_cell_enum.resize(NCells);
	// enumerate valid cells
	thrust::inclusive_scan(valid_cell_flags.begin(), valid_cell_flags.end(),
	                       valid_cell_enum.begin());
	int num_valid_cells = valid_cell_enum.back();

	std::cout << "valid cells enum: ";
	thrust::copy(valid_cell_enum.begin(), valid_cell_enum.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	std::cout << "number of valid cells: " << num_valid_cells << std::endl;

	// no valid cells at all, return with empty vertices vector.
	if (num_valid_cells == 0) {
	    vertices.clear();
	    return;
	}

	valid_cell_indices.resize(num_valid_cells);
	// generate indices to cells that pass threshold
	thrust::upper_bound(valid_cell_enum.begin(), valid_cell_enum.end(),
	                    CountingIterator(0), CountingIterator(0)+num_valid_cells,
	                    valid_cell_indices.begin());
	std::cout << "indices to valid cells: ";
	thrust::copy(valid_cell_indices.begin(), valid_cell_indices.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	// calculate how many neighbors of a cell are at the boundary of the blob.
	num_boundary_cell_neighbors.resize(num_valid_cells);
	thrust::transform(valid_cell_indices.begin(), valid_cell_indices.end(),
	                  num_boundary_cell_neighbors.begin(),
	                  boundary_cell_neighbors(input, thrust::raw_pointer_cast(&*valid_cell_flags.begin())));
	std::cout << "# of boundary cell neighbors: ";
	thrust::copy(num_boundary_cell_neighbors.begin(), num_boundary_cell_neighbors.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	// test if a cell is at the boundary of the blob.
	boundary_cell_flags.resize(num_valid_cells);
	thrust::transform(num_boundary_cell_neighbors.begin(), num_boundary_cell_neighbors.end(),
	                  boundary_cell_flags.begin(),
	                  is_boundary_cell());
	std::cout << "is boundary cell: ";
	thrust::copy(boundary_cell_flags.begin(), boundary_cell_flags.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	// enumerate how many boundary cells we have
	boundary_cell_enum.resize(num_valid_cells);
	thrust::inclusive_scan(boundary_cell_flags.begin(), boundary_cell_flags.end(),
	                       boundary_cell_enum.begin());
	std::cout << "boundary cells enum: ";
	thrust::copy(boundary_cell_enum.begin(), boundary_cell_enum.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	// total number of boundary cells
	int num_boundary_cells = boundary_cell_enum.back();
	std::cout << "number of boundary cells: " << num_boundary_cells << std::endl;

	// search for indices to boundary cells among all valid cells
	boundary_cell_indices.resize(num_boundary_cells);
	thrust::upper_bound(boundary_cell_enum.begin(), boundary_cell_enum.end(),
	                    CountingIterator(0), CountingIterator(0)+num_boundary_cells,
	                    boundary_cell_indices.begin());
	std::cout << "indices to boundary cells in valid cells: ";
	thrust::copy(boundary_cell_indices.begin(), boundary_cell_indices.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	std::cout << "indices to boundary cells in all cells: ";
	thrust::copy(thrust::make_permutation_iterator(valid_cell_indices.begin(), boundary_cell_indices.begin()),
	             thrust::make_permutation_iterator(valid_cell_indices.begin(), boundary_cell_indices.end()),
	             std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	vertices.resize(num_valid_cells*24);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(boundary_cell_indices.begin(),
	                                                              thrust::make_permutation_iterator(valid_cell_indices.begin(), boundary_cell_indices.begin()))),
	                 thrust::make_zip_iterator(thrust::make_tuple(boundary_cell_indices.end(),
	                                                              thrust::make_permutation_iterator(valid_cell_indices.begin(), boundary_cell_indices.begin()))),
	                 generate_quads(input, thrust::raw_pointer_cast(&*vertices.begin())));

    }


    // FixME: the input data type should really be cells rather than cell_ids
    struct threshold_cell : public thrust::unary_function<int, bool>
    {
	// FixME: constant iterator and/or iterator to const problem.
//	InputPointDataIterator point_data;
	float *point_data;

	const int xdim;
	const int ydim;
	const int zdim;
	const int cells_per_layer;

	const ThresholdFunction &threshold;
	__host__ __device__
	bool test(const float f) const { return (f >=4 && f <= 16); }

	threshold_cell(InputDataSet &input, ThresholdFunction threshold) :
	    point_data(thrust::raw_pointer_cast(&*input.point_data_begin())),
	    threshold(threshold),
	    xdim(input.xdim), ydim(input.ydim), zdim(input.zdim),
	    cells_per_layer((xdim - 1) * (ydim - 1)){}

	__host__ __device__
	bool operator() (int cell_id) const {
	    const int x = cell_id % (xdim - 1);
	    const int y = (cell_id / (xdim - 1)) % (ydim -1);
	    const int z = cell_id / cells_per_layer;

	    // indices to the eight vertices of the voxel
	    const int i0 = x    + y*xdim + z * xdim * ydim;
	    const int i1 = i0   + 1;
	    const int i2 = i0   + 1	+ xdim;
	    const int i3 = i0   + xdim;

	    const int i4 = i0   + xdim * ydim;
	    const int i5 = i1   + xdim * ydim;
	    const int i6 = i2   + xdim * ydim;
	    const int i7 = i3   + xdim * ydim;

	    // scalar values of the eight vertices
	    const float f0 = *(point_data + i0);
	    const float f1 = *(point_data + i1);
	    const float f2 = *(point_data + i2);
	    const float f3 = *(point_data + i3);
	    const float f4 = *(point_data + i4);
	    const float f5 = *(point_data + i5);
	    const float f6 = *(point_data + i6);
	    const float f7 = *(point_data + i7);

	    // a cell is considered passing the threshold if all of its vertices
	    // are passing the threshold.
	    bool valid = test(f0);
	    valid &= test(f1);
	    valid &= test(f2);
	    valid &= test(f3);
	    valid &= test(f4);
	    valid &= test(f5);
	    valid &= test(f6);
	    valid &= test(f7);

//	    bool valid = threshold(f0);
//	    valid &= threshold(f1);
//	    valid &= threshold(f2);
//	    valid &= threshold(f3);
//	    valid &= threshold(f4);
//	    valid &= threshold(f5);
//	    valid &= threshold(f6);
//	    valid &= threshold(f7);

//	    std::cout << "cell id: " << cell_id << ", valid: " << valid << std::endl;
	    return valid;
//	    return true;
	}
    };

    struct boundary_cell_neighbors : public thrust::unary_function<int, int>
    {
	const int xdim;
	const int ydim;
	const int zdim;
	const int cells_per_layer;

	const bool *valid_cell_flags;

	boundary_cell_neighbors(InputDataSet &input, const bool *valid_cell_flags) :
	    xdim(input.xdim), ydim(input.ydim), zdim(input.zdim),
	    cells_per_layer((xdim - 1) * (ydim - 1)),
	    valid_cell_flags(valid_cell_flags) {}

	__host__ __device__
	int operator() (int valid_cell_id) const {
	    // cell ids of the cell's six neighbors
	    const int n0 = valid_cell_id - (xdim - 1);
	    const int n1 = valid_cell_id + 1;
	    const int n2 = valid_cell_id + (xdim - 1);
	    const int n3 = valid_cell_id - 1;
	    const int n4 = valid_cell_id - cells_per_layer;
	    const int n5 = valid_cell_id + cells_per_layer;

	    // if the cell is at the boundary of the whole data set,
	    // it has a boundary cell neighbor at that face.
	    const int x = valid_cell_id % (xdim - 1);
	    const int y = (valid_cell_id / (xdim - 1)) % (ydim -1);
	    const int z = valid_cell_id / cells_per_layer;

	    int boundary = !(y == 0)          && *(valid_cell_flags + n0);
	    boundary    += !(x == (xdim - 2)) && *(valid_cell_flags + n1);
	    boundary    += !(y == (ydim - 2)) && *(valid_cell_flags + n2);
	    boundary    += !(x == 0)          && *(valid_cell_flags + n3);
	    boundary    += !(z == 0)          && *(valid_cell_flags + n4);
	    boundary    += !(z == (zdim - 2)) && *(valid_cell_flags + n5);

	    return boundary;
	}
    };

    struct is_boundary_cell : public thrust::unary_function<int, bool>
    {
	__host__ __device__
	bool operator() (int num_boundary_cell_neighbors) const {
//	    return !(num_boundary_cell_neighbors == 0 ||
//		     num_boundary_cell_neighbors != 6);
	    return num_boundary_cell_neighbors != 6;
	}
    };

    // FixME: the input data type should really be cells rather than cell_ids
    // FixME: should only generate quads for real outer/boundary faces
    struct generate_quads : public thrust::unary_function<thrust::tuple<int, int>, void>
    {
//	InputDataSet &input;
	float *point_data;
	const int xdim;
	const int ydim;
	const int zdim;
	const int cells_per_layer;

	float4 *vertices;

	generate_quads(InputDataSet &input, float4 *vertices) :
	    point_data(thrust::raw_pointer_cast(&*input.point_data_begin())),
	    xdim(input.xdim), ydim(input.ydim), zdim(input.zdim),
	    cells_per_layer((xdim - 1) * (ydim - 1)),
	    vertices(vertices) {}

	__host__ __device__
	void operator() (thrust::tuple<int, int> indices_tuple) {
	    const int valid_cell_id  = thrust::get<0>(indices_tuple);
	    const int global_cell_id = thrust::get<1>(indices_tuple);

	    const int vertices_for_faces[] =
	    {
		 0, 1, 5, 4, // face 0
		 1, 2, 6, 5, // face 1
		 2, 3, 7, 6,
		 0, 4, 7, 3,
		 0, 3, 2, 1,
		 4, 5, 6, 7
	    };

	    const int x = global_cell_id % (xdim - 1);
	    const int y = (global_cell_id / (xdim - 1)) % (ydim -1);
	    const int z = global_cell_id / cells_per_layer;

	    // indices to the eight vertices of the voxel
//	    const int i0 = x    + y*xdim + z * xdim * ydim;
//	    const int i1 = i0   + 1;
//	    const int i2 = i0   + 1	+ xdim;
//	    const int i3 = i0   + xdim;
//
//	    const int i4 = i0   + xdim * ydim;
//	    const int i5 = i1   + xdim * ydim;
//	    const int i6 = i2   + xdim * ydim;
//	    const int i7 = i3   + xdim * ydim;

	    // scalar values of the eight vertices
//	    const float f0 = *(point_data + i0);
//	    const float f1 = *(point_data + i1);
//	    const float f2 = *(point_data + i2);
//	    const float f3 = *(point_data + i3);
//	    const float f4 = *(point_data + i4);
//	    const float f5 = *(point_data + i5);
//	    const float f6 = *(point_data + i6);
//	    const float f7 = *(point_data + i7);

	    // position of the eight vertices
	    float3 p[8];
	    p[0] = make_float3(x, y, z);
	    p[1] = p[0] + make_float3(1.0f, 0.0f, 0.0f);
	    p[2] = p[0] + make_float3(1.0f, 1.0f, 0.0f);
	    p[3] = p[0] + make_float3(0.0f, 1.0f, 0.0f);
	    p[4] = p[0] + make_float3(0.0f, 0.0f, 1.0f);
	    p[5] = p[0] + make_float3(1.0f, 0.0f, 1.0f);
	    p[6] = p[0] + make_float3(1.0f, 1.0f, 1.0f);
	    p[7] = p[0] + make_float3(0.0f, 1.0f, 1.0f);

	    // FixME: should output 8 vertices/scalars with 24 indices
	    for (int v = 0; v < 24; v++) {
		*(vertices + valid_cell_id*24 + v) = make_float4(p[vertices_for_faces[v]], 1.0f);
	    }
	}
    };

    VerticesIterator verticesBegin() {
	return vertices.begin();
    }
    VerticesIterator verticesEnd() {
	return vertices.end();
    }
};
