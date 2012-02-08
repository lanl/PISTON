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
/*
 * mandelbrot_field.h
 *
 *  Created on: Feb 6, 2012
 *      Author: ollie
 */

#ifndef MANDELBROT_FIELD_H_
#define MANDELBROT_FIELD_H_

#include <piston/image2d.h>
#include <piston/choose_container.h>
#include <piston/implicit_function.h>

namespace piston {

template <typename Space>
struct mandelbrot_field : public piston::image2d<int, int, Space>
{
    typedef piston::image2d<int, int, Space> Parent;

    // transfrom (i, j) grid coordinates to shifted and scaled (x, y) position
    struct grid_coordinates_functor : public thrust::unary_function<thrust::tuple<int, int>,
								    thrust::tuple<float, float> >
    {
	float xmin;
	float ymin;
	float deltax;
	float deltay;

	grid_coordinates_functor(float xmin, float ymin, float deltax, float deltay) :
	    xmin(xmin), ymin(ymin), deltax(deltax), deltay(deltay) {}

	__host__ __device__
	thrust::tuple<float, float> operator() (thrust::tuple<int, int> grid_coord) const {
	    const float x = xmin + deltax * thrust::get<0>(grid_coord);
	    const float y = ymin + deltay * thrust::get<1>(grid_coord);
	    return thrust::make_tuple(x, y);
	}
    };

    // It is VERY IMPORTANT to declare grid_coordinates_iterator before point_data_vector
    typedef typename thrust::transform_iterator<grid_coordinates_functor,
	    typename Parent::GridCoordinatesIterator> GridCoordinatesIterator;
    GridCoordinatesIterator grid_coordinates_iterator;

    typedef typename detail::choose_container<typename Parent::CountingIterator, int>::type PointDataContainer;
    PointDataContainer point_data_vector;
    typedef typename PointDataContainer::iterator PointDataIterator;

    struct mandelbrot_functor : public piston::implicit_function2d<float, int>
    {
	typedef piston::implicit_function2d<float, int> Parent;
	typedef typename Parent::InputType InputType;
	static const int MAX_ITERS = 16;
	static const float MAX_R2 = 100.0f;

	__host__ __device__
	int operator() (InputType pos) const {
	    int iters = 0;
	    // z = (0, 0) initially
	    float x = 0;
	    float y = 0;
	    // p
	    float u = thrust::get<0>(pos);
	    float v = thrust::get<1>(pos);
	    // |z|^2
	    float r2 = x*x + y*y;

	    // iterate z = z^2 + p, |z|^2 < MAX_R2
	    while (r2 < MAX_R2 && iters++ < MAX_ITERS) {
		float x1 = x;
		x = x*x - y*y + u;
		y = 2*y*x1 + v;
		r2 = x*x + y*y;
	    }

	    return iters;
	}
    };

    mandelbrot_field(int dim0, int dim1, float xmin, float ymin, float xmax, float ymax) :
	Parent(dim0, dim1),
	grid_coordinates_iterator(Parent::grid_coordinates_iterator,
	                          grid_coordinates_functor(xmin, ymin, (xmax-xmin)/dim0, (ymax-ymin)/dim1)),
	point_data_vector(thrust::make_transform_iterator(grid_coordinates_begin(), mandelbrot_functor()),
	                  thrust::make_transform_iterator(grid_coordinates_end(),   mandelbrot_functor()))
    {}

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


#endif /* MANDELBROT_FIELD_H_ */
