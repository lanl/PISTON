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

#ifndef VTK_IMAGE3D_H_
#define VTK_IMAGE3D_H_

#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <piston/image3d.h>
#include <piston/choose_container.h>

namespace piston {

template <typename IndexType, typename ValueType, typename Space>
struct vtk_image3d : public piston::image3d<IndexType, ValueType, Space>
{
    typedef piston::image3d<IndexType, ValueType, Space> Parent;

    // transform pointid <- [0..NPoints] to grid coordinates of
    // (i <- [0..xdim-1], j <- [0..ydim-1], k <- [0..zdim-1])
    struct grid_coordinates_functor : public thrust::unary_function<IndexType, thrust::tuple<float, float, float> >
    {
        float xmin, ymin, zmin;
        float deltax, deltay, deltaz;

        grid_coordinates_functor(float xmin   = 0.0f, float ymin   = 0.0f, float zmin   = 0.0f,
                                 float deltax = 1.0f, float deltay = 1.0f, float deltaz = 1.0f) :
            xmin(xmin), ymin(ymin), zmin(zmin),
            deltax(deltax), deltay(deltay), deltaz(deltaz)
        {
//          std::cerr
//            << "GCF: "
//            << xmin << ", " << ymin << ", " << zmin << " "
//            << deltax << ", " << deltay << ", " << deltaz << std::endl;
        }

        __host__ __device__
        thrust::tuple<float, float, float> operator()(thrust::tuple<float, float, float> grid_coord) const {
            const float x = xmin + deltax * thrust::get<0>(grid_coord);
            const float y = ymin + deltay * thrust::get<1>(grid_coord);
            const float z = zmin + deltaz * thrust::get<2>(grid_coord);

            return thrust::make_tuple(x, y, z);
        }
    };

    typedef typename thrust::tuple<float, float, float> GridCoordinatesType;

    typedef typename thrust::transform_iterator<grid_coordinates_functor,
	    typename Parent::GridCoordinatesIterator> GridCoordinatesIterator;
    GridCoordinatesIterator grid_coordinates_iterator;

    typedef typename detail::choose_container<typename Parent::CountingIterator, ValueType>::type PointDataContainer;
    PointDataContainer point_data_vector;
    typedef typename PointDataContainer::iterator PointDataIterator;

    vtk_image3d(vtkImageData *image) :
	Parent(image->GetDimensions()[0], image->GetDimensions()[1], image->GetDimensions()[2]),
	grid_coordinates_iterator(Parent::grid_coordinates_iterator,
	                          grid_coordinates_functor(
	                        			   image->GetOrigin()[0]+((float)image->GetExtent()[0]*image->GetSpacing()[0]),
	                        			   image->GetOrigin()[1]+((float)image->GetExtent()[2]*image->GetSpacing()[1]),
	                        		           image->GetOrigin()[2]+((float)image->GetExtent()[4]*image->GetSpacing()[2]),
	                        		           image->GetSpacing()[0],
	                        		           image->GetSpacing()[1],
	                        		           image->GetSpacing()[2])),
	point_data_vector((ValueType *) image->GetScalarPointer(),
	                  (ValueType *) image->GetScalarPointer() + this->NPoints)
    {
//	std::cout << "Origin: ";
//	for (int i = 0; i < 3; i++) {
//	    std::cout << image->GetOrigin()[i] << ", ";
//	}
//	std::cout << std::endl;
//	std::cout << "Extent: ";
//	for (int i = 0; i < 6; i++) {
//	    std::cout << image->GetExtent()[i] << ", ";
//	}
//	std::cout << std::endl;
//	std::cout << "Spacing: ";
//	for (int i = 0; i < 3; i++) {
//	    std::cout << image->GetSpacing()[i] << ", ";
//	}
//	std::cout << std::endl;
    }
#if 0
	point_data_vector((ValueType *) image->GetPointData()->GetArray("Elevation")->GetVoidPointer(0),
	                  (ValueType *) image->GetPointData()->GetArray("Elevation")->GetVoidPointer(0) + this->NPoints) {
	}
#endif

#if 0
    vtk_image3d(int xdim, int ydim, int zdim,
                float xmin = 0.0f, float ymin = 0.0f, float zmin = 0.0f,
                float deltax = 1.0f, float deltay = 1.0f, float deltaz = 1.0f) :
        Parent(xdim, ydim, zdim),
	grid_coordinates_iterator(Parent::grid_coordinates_iterator,
	                          grid_coordinates_functor(xmin, ymin, zmin, deltax, deltay, deltaz)),
	point_data_vector(this->NPoints) {}
#endif

#if 1

    // TODO: COPY Constructor??, Constructor from another image/vector?
    vtk_image3d(int dims[3], thrust::device_vector<ValueType> v) :
        Parent(dims[0], dims[1], dims[2]),
//        grid_coordinates_vector(Parent::grid_coordinates_begin(), Parent::grid_coordinates_end()),
        grid_coordinates_iterator(Parent::grid_coordinates_iterator,
        	                          grid_coordinates_functor(0,0,0, 1,1,1)),
        point_data_vector(v.begin(), v.end()) {
	std::cout << "copy constructor" << std::endl;
    }
#endif

    void resize(int xdim, int ydim, int zdim) {
 	Parent::resize(xdim, ydim, zdim);
 	// TBD, is there resize in VTK?
     }


    GridCoordinatesIterator grid_coordinates_begin() {
	return grid_coordinates_iterator;
    }
    GridCoordinatesIterator grid_coordinates_end() {
	return grid_coordinates_iterator+this->NPoints;
    }

    PointDataIterator point_data_begin() {
	return point_data_vector.begin();
    }
    PointDataIterator point_data_end() {
	return point_data_vector.end();
    }
};

}
#endif /* VTK_IMAGE3D_H_ */
