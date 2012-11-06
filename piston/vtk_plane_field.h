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

#ifndef VTK_PLANE_FIELD_H_
#define VTK_PLANE_FIELD_H_

#include <piston/choose_container.h>
#include <piston/util/plane_functor.h>
#include <piston/image3d.h>

namespace piston {

template <typename IndexType, typename ValueType, typename Space>
struct vtk_plane_field : public piston::image3d<Space>
{
    ValueType origin[3];
    ValueType spacing[3];
    ValueType extents[6];

    struct grid_coordinates_functor : public thrust::unary_function<
        IndexType, thrust::tuple<ValueType, ValueType, ValueType> > {
      int xdim;
      int ydim;
      int zdim;
      int PointsPerLayer;
      ValueType xmin, ymin, zmin;
      ValueType deltax, deltay, deltaz;

      grid_coordinates_functor(
          int dimx, int dimy, int dimz,
          ValueType minx = 0.0f, ValueType miny = 0.0f, ValueType minz = 0.0f,
          ValueType dx = 0.0f, ValueType dy = 0.0f, ValueType dz = 0.0f) :
          xdim(dimx), ydim(dimy), zdim(dimz),
          PointsPerLayer(xdim*ydim),
          xmin(minx), ymin(miny), zmin(minz),
          deltax(dx), deltay(dy), deltaz(dz)
      {
      }

      __host__ __device__
      thrust::tuple<ValueType, ValueType, ValueType> operator()(
          IndexType PointId) const
      {
          const ValueType x = xmin + deltax * (PointId % xdim);
          const ValueType y = ymin + deltay * ((PointId/xdim) % ydim);
          const ValueType z = zmin + deltaz * (PointId/PointsPerLayer);

          return thrust::make_tuple(x,y,z);
      }
    };

    typedef typename thrust::counting_iterator<IndexType, Space>
        CountingIterator;
    typedef typename thrust::transform_iterator<grid_coordinates_functor,
        CountingIterator> GridCoordinatesTransformIterator;

    GridCoordinatesTransformIterator grid_coordinates_iterator;

    typedef piston::image3d<Space> Parent;

    typedef typename detail::choose_container<typename Parent::CountingIterator,
        thrust::tuple<ValueType, ValueType, ValueType> >::type
          GridCoordinatesContainer;
    GridCoordinatesContainer grid_coordinates_vector;

    typedef typename GridCoordinatesContainer::iterator
        GridCoordinatesIterator;
    typedef typename detail::choose_container<typename Parent::CountingIterator,
      ValueType>::type PointDataContainer;
    PointDataContainer point_data_vector;

    typedef typename PointDataContainer::iterator PointDataIterator;

    vtk_plane_field(ValueType origin[3], ValueType normal[3], int dims[3],
                    ValueType spacing[3], int extents[6]) :
          Parent(dims[0], dims[1], dims[2]),
          grid_coordinates_iterator(thrust::make_transform_iterator(
              CountingIterator(0),
              grid_coordinates_functor(
                  dims[0], dims[1], dims[2],
                  origin[0] + ((ValueType)extents[0] * spacing[0]),
                  origin[1] + ((ValueType)extents[2] * spacing[1]),
                  origin[2] + ((ValueType)extents[4] * spacing[2]),
                  spacing[0], spacing[1], spacing[2])
              )
          ),
          grid_coordinates_vector(grid_coordinates_iterator,
                                  grid_coordinates_iterator + Parent::NPoints),
          point_data_vector(
              thrust::make_transform_iterator(grid_coordinates_vector.begin(),
              plane_functor<IndexType, ValueType>(
                  make_float3(origin[0], origin[1], origin[2]),
                  make_float3(normal[0], normal[1], normal[2]))),
              thrust::make_transform_iterator(
                  grid_coordinates_vector.end(),
                  plane_functor<IndexType, ValueType>(
                      make_float3(origin[0], origin[1], origin[2]),
                      make_float3(normal[0], normal[1], normal[2]))))
        {
            this->origin[0] = origin[0];
            this->origin[1] = origin[1];
            this->origin[2] = origin[2];

            this->spacing[0] = spacing[0];
            this->spacing[1] = spacing[1];
            this->spacing[2] = spacing[2];

            this->extents[0] = extents[0];
            this->extents[1] = extents[1];
            this->extents[2] = extents[2];
            this->extents[3] = extents[3];
            this->extents[4] = extents[4];
            this->extents[5] = extents[5];
        }

    void resize(ValueType origin[3], ValueType normal[3], int xdim, int ydim, int zdim) {
        Parent::resize(xdim, ydim, zdim);

        grid_coordinates_vector.resize(this->NPoints);
        grid_coordinates_vector.assign(Parent::grid_coordinates_begin(),
                                       Parent::grid_coordinates_end());

        point_data_vector.resize(this->NPoints);
        point_data_vector.assign(
            thrust::make_transform_iterator(
                grid_coordinates_vector.begin(),
                plane_functor<IndexType, ValueType>(
                    make_float3(origin[0], origin[1], origin[2]),
                    make_float3(normal[0], normal[1], normal[2]))),
            thrust::make_transform_iterator(
                grid_coordinates_vector.end(),
                plane_functor<IndexType, ValueType>(
                    make_float3(origin[0], origin[1], origin[2]),
                    make_float3(normal[0], normal[1], normal[2]))));
    }

    GridCoordinatesIterator grid_coordinates_begin() {
        return grid_coordinates_vector.begin();
    }

    GridCoordinatesIterator grid_coordinates_end() {
        return grid_coordinates_vector.end();
    }

    PointDataIterator point_data_begin() {
        return point_data_vector.begin();
    }

    PointDataIterator point_data_end() {
        return point_data_vector.end();
    }

private:
    vtk_plane_field(const vtk_plane_field&);
    vtk_plane_field& operator=(const vtk_plane_field&);
};

}

#endif /* VTK_PLANE_FIELD_H_ */
