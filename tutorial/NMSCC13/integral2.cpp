/*
 * integral2.cpp
 *
 *  Copyright (c) 2013, Los Alamos National Security, LLC.
 *  All rights Reserved.
 *
 *  Copyright 2013. Los Alamos National Security, LLC. This software was produced
 *  under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National
 *  Laboratory (LANL), which is operated by Los Alamos National Security, LLC
 *  for the U.S. Department of Energy. The U.S. Government has rights to use,
 *  reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS
 *  ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 *  ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified
 *  to produce derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Los Alamos National Security, LLC, Los Alamos
 *       National Laboratory, LANL, the U.S. Government, nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 *  NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL
 *  SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Oct 6, 2013
 *      Author: ollie
 *
 *  Simple numerical integration with scan
 */

#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/iterator/constant_iterator.h>

#include "utils.h"

// we can change how often we want to sample the function f(x) = x^2 by
// changing the constant N.
const int    N = 10;
const float dx = 1.0f/N;

// Unary functor for the quaddratic function, this functor takes one floating
// point number x and return the square of x. 
struct square : public thrust::unary_function<float, float>
{
    __host__ __device__
    float
    operator() (float x) const {
	return x*x;
    }
};

int
main()
{
    // allocate a vector for a sequence of floats, x_i
    thrust::device_vector<float> x(N+1);

    // generate a sequence of x_i in [0, 1] with dx in between each of them.
    thrust::sequence(x.begin(), x.end(), 0.0f, dx);

    // print the x_i
    std::cout << "x:\t\t";
    std::for_each(x.begin(), x.end(), print_float(6));
    std::cout << std::endl;

    my_pause();

    // allocate a vector for f(x_i) = y_i = x_i^2.
    thrust::device_vector<float> y(N+1);

    // transform x_i into f(x_i) = y_i =  x_i^2
    thrust::transform(x.begin(), x.end(),
                      y.begin(),
                      square());

    // print the y_i
    std::cout << "y = x^2:\t";
    std::for_each(y.begin(), y.end(), print_float(6));
    std::cout << std::endl;

    my_pause();

    // print the constant dx
    std::cout << "dx:\t\t";
    std::for_each(thrust::constant_iterator<float>(dx),
                  thrust::constant_iterator<float>(dx)+N+1,
                  print_float(6));
    std::cout << std::endl;

    my_pause();

    // allocate a vector for f(x_i) * dx = y_i * dx
    thrust::device_vector<float> y_dx(N+1);

    // multiply f(x_i) by dx => y_i * dx
    thrust::transform(y.begin(), y.end(),
                      thrust::constant_iterator<float>(dx),
                      y_dx.begin(),
                      thrust::multiplies<float>());

    // print y_i * dx;
    std::cout << "y * dx:\t\t";
    std::for_each(y_dx.begin(), y_dx.end(), print_float(6));
    std::cout << std::endl;

    my_pause();

    // alocate a vector for F(t) = integrate(f(x) * dx)
    thrust::device_vector<float> F(N+1);

    // we use a scan to have a running accumulattion of f(t) dt
    thrust::inclusive_scan(y_dx.begin(), y_dx.end(),
                           F.begin());

    std::cout << "scan(y * dx, +):";
    std::for_each(F.begin(), F.end(), print_float(6));
    std::cout << std::endl;
}
