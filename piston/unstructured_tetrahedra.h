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

#ifndef UNSTRUCTURED_TETRAHEDRA_H_
#define UNSTRUCTURED_TETRAHEDRA_H_

#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <piston/image3d.h>
#include <piston/choose_container.h>

namespace piston {

// TODO: inherit from image2d?
template <typename MemorySpace>
struct unstructured_tetrahedra
{
    typedef unsigned IndexType;

    IndexType NPoints;
    IndexType NCells;

    thrust::device_vector<float> raw_data; 
    thrust::device_vector<vtkIdType> cell_array;
    thrust::device_vector<float> vertex_array;

    struct grid_coordinates_functor : public thrust::unary_function<IndexType, thrust::tuple<float, float, float> >
    {
        vtkIdType* cdata;
        float* vdata;
        float sfactor;

	grid_coordinates_functor(float sfactor, vtkIdType* cdata, float* vdata) : sfactor(sfactor), cdata(cdata), vdata(vdata) {}

	__host__ __device__
	thrust::tuple<float, float, float>
	operator()(const IndexType& point_id) const {
            vtkIdType cellId = point_id / 4;
            int vertexId = point_id % 4;
            int vindex = cdata[5*cellId+vertexId+1];   
            return thrust::make_tuple(vdata[vindex*3]*sfactor, vdata[vindex*3+1]*sfactor, vdata[vindex*3+2]*sfactor);
	}
    };

    
    struct point_data_functor : public thrust::unary_function<IndexType, float>
    {
        float* rdata;
        vtkIdType* cdata;

	point_data_functor(float* rdata, vtkIdType* cdata) : rdata(rdata), cdata(cdata) {}

	__host__ __device__
	float operator()(const IndexType& point_id) const {
            vtkIdType cellId = point_id / 4;
            int vertexId = point_id % 4;
            int vindex = cdata[5*cellId+vertexId+1];
            return (rdata[vindex]);
	}
    };

    typedef typename thrust::counting_iterator<IndexType, MemorySpace> CountingIterator;
    typedef typename thrust::transform_iterator<grid_coordinates_functor, CountingIterator> GridCoordinatesIterator;
    GridCoordinatesIterator grid_coordinates_iterator;

    typedef typename thrust::transform_iterator<point_data_functor, CountingIterator> PointDataIterator;
    PointDataIterator point_data_iterator;

    unstructured_tetrahedra(vtkUnstructuredGrid* ugrid, float scale_factor=1.0f) :
	NPoints(ugrid->GetNumberOfPoints()),
	NCells(ugrid->GetNumberOfCells()),
        raw_data((float*)(vtkFloatArray::SafeDownCast(ugrid->GetPointData()->GetArray(0))->GetPointer(0)),
                 (float*)(vtkFloatArray::SafeDownCast(ugrid->GetPointData()->GetArray(0))->GetPointer(0)+ugrid->GetNumberOfPoints())),
        cell_array((vtkIdType*)(ugrid->GetCells()->GetPointer()), (vtkIdType*)(ugrid->GetCells()->GetPointer()+5*ugrid->GetNumberOfCells())),
        vertex_array((float*)(vtkFloatArray::SafeDownCast(ugrid->GetPoints()->GetData())->GetPointer(0)),
                     (float*)(vtkFloatArray::SafeDownCast(ugrid->GetPoints()->GetData())->GetPointer(0)+3*ugrid->GetNumberOfPoints())),
	grid_coordinates_iterator(CountingIterator(0), grid_coordinates_functor(scale_factor, thrust::raw_pointer_cast(&*cell_array.begin()), thrust::raw_pointer_cast(&*vertex_array.begin()))),
        point_data_iterator(CountingIterator(0), point_data_functor(thrust::raw_pointer_cast(&*raw_data.begin()), thrust::raw_pointer_cast(&*cell_array.begin())))  {}

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

    PointDataIterator point_data_begin() {
	return point_data_iterator;
    }
    PointDataIterator point_data_end() {
	return point_data_iterator+NCells*4; 
    }
};

}
#endif /* UNSTRUCTURED_TETRAHEDRA_H_ */







