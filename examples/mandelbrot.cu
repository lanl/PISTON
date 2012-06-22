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
 * mandelbrot.cu
 *
 * Text based Mandelbrot Set visualizer inspired by the "Functional Pearl: Composing fractals, Mark P. Jones"
 * The piston::mandelbrot_field is essentially a map between grid_point_id::[Int] to ASCII Image::[Char]
 *
 *  Created on: Feb 7, 2012
 *      Author: ollie
 */
#include <iostream>
#include <piston/util/mandelbrot_field.h>

#define SPACE thrust::detail::default_device_space_tag
//#define SPACE thrust::host_space_tag

int main(int argc, char *argv[])
{
    char pallette[] = " ,.\'\\\"~:;o-!|?/<>X+={^0#&@8*$";

    int ncols = 80;
    int nrows = 40;
    float xmin = -3.0f;
    float ymin = -2.0f;
    float xmax =  2.0f;
    float ymax =  2.0f;

    piston::mandelbrot_field<SPACE> mandelbrot(ncols, nrows, xmin, ymin, xmax, ymax);
    piston::mandelbrot_field<SPACE>::PointDataIterator iter = mandelbrot.point_data_begin();

    for (int row = nrows-1; row >= 0; row--) {
	for (int col = 0; col < ncols; col++) {
	    std::cout << pallette[iter[row*ncols + col]];
	}
	std::cout << std::endl;
    }
}
