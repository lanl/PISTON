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

/* Terminologies:
 * 	Boundary: cells at the borders of the input dataset
 * 	Valid   : cells with all the vertices passing the threshold
 * 	Invalid : cells with at least one vertices failing the threshold
 * 	Exterior valid : valid cells that generate geometry
 * 	Interior valid : valid cells that don't generate geometry */
template <typename InputDataSet>
struct threshold_geometry
{
    typedef typename InputDataSet::PointDataIterator InputPointDataIterator;
    typedef typename InputDataSet::GridCoordinatesIterator InputGridCoordinatesIterator;

    typedef typename thrust::iterator_difference<InputPointDataIterator>::type	diff_type;
    typedef typename thrust::iterator_space<InputPointDataIterator>::type	space_type;
    typedef typename thrust::iterator_value<InputPointDataIterator>::type	value_type;

    typedef typename thrust::counting_iterator<int, space_type>	CountingIterator;

    typedef typename choose_container<InputPointDataIterator, int>::type  IndicesContainer;
    typedef typename choose_container<InputPointDataIterator, int>::type ValidFlagsContainer;

    typedef typename IndicesContainer::iterator IndicesIterator;
    typedef typename ValidFlagsContainer::iterator ValidFlagsIterator;
    typedef thrust::permutation_iterator<InputGridCoordinatesIterator, IndicesIterator> VerticesIterator;
    typedef thrust::permutation_iterator<InputPointDataIterator, IndicesIterator> ScalarsIterator;

    InputDataSet &input;
    float min_value;
    float max_value;

    ValidFlagsContainer valid_cell_flags;
    IndicesContainer    valid_cell_enum;
    IndicesContainer    valid_cell_indices;
    IndicesContainer    num_valid_cell_neighbors;
    ValidFlagsContainer exterior_cell_flags;
    IndicesContainer    exterior_cell_enum;
    IndicesContainer    exterior_cell_indices;
    IndicesContainer	vertices_indices;

    threshold_geometry(InputDataSet &input,float min_value, float max_value ) :
	input(input), min_value(min_value), max_value(max_value) {}

    void operator()() {
	const int NCells = input.NCells;

//	std::cout << std::endl;

	valid_cell_flags.resize(NCells);

	// test and enumerate cells that pass threshold, we don't do kernel fusion
	// because the flags are used in a later stage.
	thrust::transform(CountingIterator(0), CountingIterator(0)+NCells,
	                  valid_cell_flags.begin(),
	                  threshold_cell(input, min_value, max_value));
//	thrust::host_vector<bool> dummy(valid_cell_flags.begin(), valid_cell_flags.end());
//	thrust::copy(dummy.begin(), dummy.end(), std::ostream_iterator<bool>(std::cout, " "));
//	std::cout << std::endl;

	valid_cell_enum.resize(NCells);
	// enumerate valid cells
	thrust::inclusive_scan(valid_cell_flags.begin(), valid_cell_flags.end(),
	                       valid_cell_enum.begin());
	int num_valid_cells = valid_cell_enum.back();

//	std::cout << "valid cells enum: ";
//	thrust::copy(valid_cell_enum.begin(), valid_cell_enum.end(), std::ostream_iterator<int>(std::cout, " "));
//	std::cout << std::endl;

//	std::cout << "number of valid cells: " << num_valid_cells << std::endl;

	// no valid cells at all, return with empty vertices vector.
	if (num_valid_cells == 0) {
	    vertices_indices.clear();
	    return;
	}

	valid_cell_indices.resize(num_valid_cells);
	// generate indices to cells that pass threshold
	thrust::upper_bound(valid_cell_enum.begin(), valid_cell_enum.end(),
	                    CountingIterator(0), CountingIterator(0)+num_valid_cells,
	                    valid_cell_indices.begin());
//	std::cout << "indices to valid cells: ";
//	thrust::copy(valid_cell_indices.begin(), valid_cell_indices.end(), std::ostream_iterator<int>(std::cout, " "));
//	std::cout << std::endl;

	// calculate how many neighbors of a cell are valid.
	num_valid_cell_neighbors.resize(num_valid_cells);
	thrust::transform(valid_cell_indices.begin(), valid_cell_indices.end(),
	                  num_valid_cell_neighbors.begin(),
	                  valid_cell_neighbors(input, thrust::raw_pointer_cast(&*valid_cell_flags.begin())));
//	std::cout << "# of valid cell neighbors: ";
//	thrust::copy(num_valid_cell_neighbors.begin(), num_valid_cell_neighbors.end(), std::ostream_iterator<int>(std::cout, " "));
//	std::cout << std::endl;

	// test if a cell is at the exterior of the blob of valid cells.
	exterior_cell_flags.resize(num_valid_cells);
	thrust::transform(num_valid_cell_neighbors.begin(), num_valid_cell_neighbors.end(),
	                  exterior_cell_flags.begin(),
	                  is_exterior_cell());
//	std::cout << "is exterior cell: ";
//	thrust::copy(exterior_cell_flags.begin(), exterior_cell_flags.end(), std::ostream_iterator<int>(std::cout, " "));
//	std::cout << std::endl;

	// enumerate how many exterior cells we have
	exterior_cell_enum.resize(num_valid_cells);
	thrust::inclusive_scan(exterior_cell_flags.begin(), exterior_cell_flags.end(),
	                       exterior_cell_enum.begin());
//	std::cout << "exteriro cells enum: ";
//	thrust::copy(exterior_cell_enum.begin(), exterior_cell_enum.end(), std::ostream_iterator<int>(std::cout, " "));
//	std::cout << std::endl;

	// total number of exterior cells
	int num_exterior_cells = exterior_cell_enum.back();
//	std::cout << "number of exterior cells: " << num_exterior_cells << std::endl;

	// search for indices to exterior cells among all valid cells
	exterior_cell_indices.resize(num_exterior_cells);
	thrust::upper_bound(exterior_cell_enum.begin(), exterior_cell_enum.end(),
	                    CountingIterator(0), CountingIterator(0)+num_exterior_cells,
	                    exterior_cell_indices.begin());
//	std::cout << "indices to exterior cells in valid cells: ";
//	thrust::copy(exterior_cell_indices.begin(), exterior_cell_indices.end(), std::ostream_iterator<int>(std::cout, " "));
//	std::cout << std::endl;

//	std::cout << "indices to exterior cells in all cells: ";
//	thrust::copy(thrust::make_permutation_iterator(valid_cell_indices.begin(), exterior_cell_indices.begin()),
//	             thrust::make_permutation_iterator(valid_cell_indices.begin(), exterior_cell_indices.end()),
//	             std::ostream_iterator<int>(std::cout, " "));
//	std::cout << std::endl;

	// generate 6 quards for each exterior cell
	vertices_indices.resize(num_exterior_cells*24);
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(CountingIterator(0),
	                                                              thrust::make_permutation_iterator(valid_cell_indices.begin(), exterior_cell_indices.begin()))),
	                 thrust::make_zip_iterator(thrust::make_tuple(CountingIterator(0) + num_exterior_cells,
	                                                              thrust::make_permutation_iterator(valid_cell_indices.begin(), exterior_cell_indices.begin()))),
	                 generate_quads(input, thrust::raw_pointer_cast(&*vertices_indices.begin())));

    }


    // FixME: the input data type should really be cells rather than cell_ids
    // FixME: change float to value_type
    struct threshold_cell : public thrust::unary_function<int, bool>
    {
	// FixME: constant iterator and/or iterator to const problem.
	InputPointDataIterator point_data;
	const float min_value;
	const float max_value;

	const int xdim;
	const int ydim;
	const int zdim;
	const int cells_per_layer;

	__host__ __device__
	bool threshold(float val) const {
	    return (min_value <= val) && (val <= max_value);
	}

	threshold_cell(InputDataSet &input, float min_value, float max_value) :
	    point_data(input.point_data_begin()),
	    min_value(min_value), max_value(max_value),
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
	    bool valid = threshold(f0);
	    valid &= threshold(f1);
	    valid &= threshold(f2);
	    valid &= threshold(f3);
	    valid &= threshold(f4);
	    valid &= threshold(f5);
	    valid &= threshold(f6);
	    valid &= threshold(f7);

	    return valid;
	}
    };

    // return the number of neighbors that are valid cells
    struct valid_cell_neighbors : public thrust::unary_function<int, int>
    {
	const int xdim;
	const int ydim;
	const int zdim;
	const int cells_per_layer;

	const int *valid_cell_flags;

	valid_cell_neighbors(InputDataSet &input, const int *valid_cell_flags) :
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

	    // we are using fixed boundary conditions here, if a cell is at
	    // the border of the data set, it DOES NOT have an valid cell
	    // neighbor at that face.
	    const int x = valid_cell_id % (xdim - 1);
	    const int y = (valid_cell_id / (xdim - 1)) % (ydim -1);
	    const int z = valid_cell_id / cells_per_layer;

	    // we are taking advantage of short-circuit && operator here,
	    // the coordinates of the cell is tested first so we won't
	    // access to data past boundary.
	    int boundary = !(y == 0)          && *(valid_cell_flags + n0);
	    boundary    += !(x == (xdim - 2)) && *(valid_cell_flags + n1);
	    boundary    += !(y == (ydim - 2)) && *(valid_cell_flags + n2);
	    boundary    += !(x == 0)          && *(valid_cell_flags + n3);
	    boundary    += !(z == 0)          && *(valid_cell_flags + n4);
	    boundary    += !(z == (zdim - 2)) && *(valid_cell_flags + n5);

	    return boundary;
	}
    };

    // return true if the cell will actually generate geometry.
    struct is_exterior_cell : public thrust::unary_function<int, bool>
    {
	__host__ __device__
	bool operator() (int num_boundary_cell_neighbors) const {
//	    return !(num_boundary_cell_neighbors == 0 ||
//		     num_boundary_cell_neighbors != 6);
	    return num_boundary_cell_neighbors != 6;
	}
    };

    // FixME: the input data type should really be cells rather than cell_ids
    struct generate_quads : public thrust::unary_function<thrust::tuple<int, int>, void>
    {
	const int xdim;
	const int ydim;
	const int zdim;
	const int cells_per_layer;

	// crazy C++ const correctness, vertices_indices is a pointer that
	// does not change the address it points to.
	int * const vertices_indices;

	generate_quads(InputDataSet &input, int * const vertices_indices) :
	    xdim(input.xdim), ydim(input.ydim), zdim(input.zdim),
	    cells_per_layer((xdim - 1) * (ydim - 1)),
	    vertices_indices(vertices_indices) {}

	__host__ __device__
	void operator() (thrust::tuple<int, int> indices_tuple) const {
	    const int exterior_cell_id = thrust::get<0>(indices_tuple);
	    const int global_cell_id   = thrust::get<1>(indices_tuple);

//	    std::cout << "exterior id: " << exterior_cell_id << ", global_cell_id: " << global_cell_id << std::endl;
	    const int vertices_for_faces[] =
	    {
		 0, 1, 5, 4, // face 0
		 1, 2, 6, 5, // face 1
		 2, 3, 7, 6, // face 2
		 0, 4, 7, 3, // face 3
		 0, 3, 2, 1, // face 4
		 4, 5, 6, 7  // face 5
	    };

	    const int x = global_cell_id % (xdim - 1);
	    const int y = (global_cell_id / (xdim - 1)) % (ydim -1);
	    const int z = global_cell_id / cells_per_layer;

	    // indices to the eight vertices of the voxel
	    int i[8];
	    i[0] = x      + y*xdim + z * xdim * ydim;
	    i[1] = i[0]   + 1;
	    i[2] = i[0]   + 1	+ xdim;
	    i[3] = i[0]   + xdim;

	    i[4] = i[0]   + xdim * ydim;
	    i[5] = i[1]   + xdim * ydim;
	    i[6] = i[2]   + xdim * ydim;
	    i[7] = i[3]   + xdim * ydim;

	    for (int v = 0; v < 24; v++) {
		*(vertices_indices + exterior_cell_id*24 + v) = i[vertices_for_faces[v]];
	    }
	}
    };


    // FixME: better name!!!
    VerticesIterator vertices_begin() {
	return thrust::make_permutation_iterator(input.grid_coordinates_begin(), vertices_indices.begin());
    }
    VerticesIterator vertices_end() {
	return thrust::make_permutation_iterator(input.grid_coordinates_begin(), vertices_indices.begin()) +
		vertices_indices.size();
    }

    // FixME: better name!!!
    ScalarsIterator scalars_begin() {
	return thrust::make_permutation_iterator(input.point_data_begin(), vertices_indices.begin());
    }
    ScalarsIterator scalars_end() {
	return thrust::make_permutation_iterator(input.point_data_begin(), vertices_indices.begin()) +
		vertices_indices.size();
    }
};
