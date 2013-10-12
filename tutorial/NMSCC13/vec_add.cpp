/*
 * vec_add.cpp
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
 *  Simple example of adding two integer vectors
 */

#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "utils.h"

int
main()
{
    // Allocate and initialize two vectors on the device. the vector x is
    // initialized with 1 while vector y is unitialized.
    thrust::device_vector<int> x(10, 1);
    thrust::device_vector<int> y(10);

    // Print out each element of x by applying the "print_integer" operator.
    std::cout << "x:\t";
    std::for_each(x.begin(), x.end(), print_integer(4));
    std::cout << std::endl;

    my_pause();

    // initialize vector y with a sequence of numbers starting from 5 with 2
    // as increment step.
    thrust::sequence(y.begin(), y.end(), 5, 2);

    std::cout << "y:\t";
    std::for_each(y.begin(), y.end(), print_integer(4));
    std::cout << std::endl;

    my_pause();

    // allocate a device vector for the result of the addition of the two
    // vectors. the content of the vector is not initialized.
    thrust::device_vector<int> result(10);

    // apply the integer addition operation to the vectors.
    thrust::transform(x.begin(), x.end(),   	// begin and end of the first input vector
                      y.begin(),            	// begin of the second input vector
                      result.begin(),		// begin of the result vector
                      thrust::plus<int>()); 	// integer addition

    // allocate a vector on the host and initialize it with data from the
    // device vector.
    thrust::host_vector<int> h_result(10);
    thrust::copy(result.begin(), result.end(),
                 h_result.begin());

    std::cout << "transform: x + y:\t";
    std::for_each(h_result.begin(), h_result.end(), print_integer(4));
    std::cout << std::endl;
}
