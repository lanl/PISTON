/*
 * fibonacci.cpp
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
 *  Created on: Oct 6, 2010
 *      Author: ollie
 *
 *  Calculate Fibonacci sequence by matrix multiplication
 */

#include <iostream>
#include <algorithm>
#include <iterator>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>

// A simple data structure for 2x2 matrix
template <typename T>
struct mat22 {
    T m00, m10, m01, m11;

    __host__ __device__
    mat22() {}

    /*     
     * m = |a c|
     *     |b d|
     */
    __host__ __device__
    mat22(T a, T b, T c, T d) : m00(a), m10(b), m01(c), m11(d) {}
};

// Overload the multiplication operator for 2x2 matrices
template <typename T>
__host__ __device__
mat22<T>
operator*(const mat22<T> &lhs, const mat22<T> &rhs)
{
    return mat22<T>(
	    lhs.m00*rhs.m00+lhs.m01*rhs.m10,
	    lhs.m10*rhs.m00+lhs.m11*rhs.m10,
	    lhs.m00*rhs.m01+lhs.m01*rhs.m11,
	    lhs.m10*rhs.m01+lhs.m11*rhs.m11);
}

// Wrap the overloaded multiplication operator into a binary functor.
template <typename T>
struct matmul : public thrust::binary_function<mat22<T>, mat22<T>, mat22<T> > {
    __host__ __device__
    mat22<T> operator()(const mat22<T> &lhs, const mat22<T> &rhs) {
	return lhs*rhs;
    }
};

// Extract the (0, 0) element from the 2x2 matrix
template <typename T>
struct zeroth_elem : public thrust::unary_function<mat22<T>, T> {
    __host__ __device__
    T operator()(const mat22<T> &mat) {
	return mat.m00;
    }
};

template <typename T>
struct print_mat22   
{
    void operator() (const mat22<T>& mat) {
        std::cout << mat.m00 << " " << mat.m01 << std::endl;
        std::cout << mat.m10 << " " << mat.m11 << std::endl;
        std::cout << std::endl;
    }
};   
                             
                             
int main(void)
{
    const int N = 20;

    mat22<unsigned long> A(0, 1, 1, 1);

    thrust::constant_iterator<mat22<unsigned long> > begin(A);
    thrust::device_vector<mat22<unsigned long> > vect(N);

    // we multiply a bunch of matrix A toghther.
    thrust::inclusive_scan(begin, begin + N,
                           vect.begin(),
                           matmul<unsigned long>());

    // extract the (0,0) element from the matrices
    thrust::device_vector<unsigned long> fib(N);
    thrust::transform(vect.begin(), vect.end(),
                      fib.begin(),
                      zeroth_elem<unsigned long>());
    
    std::cout << "The first " << N << " Fibonacci numbers: ";
    thrust::copy(fib.begin(), fib.end(),
                 std::ostream_iterator<unsigned long>(std::cout, " "));
    std::cout << std::endl;
}
