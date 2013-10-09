/*
 * primes.cpp
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
 *  Created on: Sept. 17, 2013
 *      Author: csewell
 *
 * Find all primes up to N using the Sieve of Eratosthenes
 */

#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

#include "utils.h"

// Find all primes up to this value
#define N 100

// Return whether a value is not divisible by a reference value
struct is_not_divisible : public thrust::unary_function<int, bool> {
    int p;

    is_not_divisible(int p) : p(p) {};

    __host__ __device__
    bool operator()(int v) {
	return ((v % p) != 0);
    }
};

int main(void)
{
    // Use two vectors, since copy_if cannot output to same vector
    // as its input
    thrust::device_vector<int> values1(N-1), values2(N-1);

    // Start with a counting iterator from 2 to N
    thrust::sequence(values1.begin(), values1.end(), 2); 
    values2[0] = 2;

    // Initialize loop variables
    thrust::device_vector<int>::iterator input_iter  = values1.begin(); 
    thrust::device_vector<int>::iterator output_iter = values2.begin(); 
    int length = values1.size();
    int primes_found = 0;

    std::cout << "Integers from 2 to " << N << ":" << std::endl;
    std::for_each(input_iter, input_iter+length, print_integer(4));
    std::cout << std::endl;

    pause();

    // Loop until we have eliminated all non-primes from the vector
    while (length > primes_found) {
	std::cout << "Eliminate all multiples of " << *(input_iter+primes_found)
		  << ":" << std::endl;

        // Use a copy_if for stream compaction to eliminate all values in
        // the vector that are divisible by the most recently found prime 
        thrust::device_vector<int>::iterator last_iter =
            thrust::copy_if(input_iter +primes_found+1, input_iter+length,
                            output_iter+primes_found+1,
                            is_not_divisible(*(input_iter+primes_found))); 
        length = last_iter - output_iter;

        std::for_each(output_iter, last_iter, print_integer(4));
        std::cout << std::endl;

        pause();

        // Swap the input and output iterators (pointers) so that the next
        // loop iteration will use the output of this iteration as its input
        thrust::swap(input_iter, output_iter);
        primes_found++;
    }

    // After all non-primes have been eliminated, we are left with only primes
    // in the vector
    std::cout << "Primes up to " << N << ": ";
    std::for_each(input_iter, input_iter+length, print_integer(4));
    std::cout << std::endl;
}
