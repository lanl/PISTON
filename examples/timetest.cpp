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

#include <piston/choose_container.h>

#define SPACE thrust::detail::default_device_space_tag
using namespace piston;

#include <piston/implicit_function.h>
#include <piston/marching_cube.h>
#include <piston/util/tangle_field.h>
#include <piston/util/plane_field.h>
#include <piston/util/sphere_field.h>
#include <piston/threshold_geometry.h>

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

struct timeval begin, end, diff;
int frame_count = 1;
int grid_size = 256;

tangle_field<int, float, SPACE>* tangle;
marching_cube<tangle_field<int, float, SPACE>, tangle_field<int, float, SPACE> > *isosurface;

plane_field<int, float, SPACE>* plane;
marching_cube<plane_field<int, float, SPACE>, tangle_field<int, float, SPACE> > *cutplane;

sphere_field<int, float, SPACE>* scalar_field;
threshold_geometry<sphere_field<int, float, SPACE> >* threshold;


int main(int argc, char** argv)
{
#ifdef TANGLE
    tangle = new tangle_field<int, float, SPACE>(grid_size, grid_size, grid_size);
    isosurface = new marching_cube<tangle_field<int, float, SPACE>,  tangle_field<int, float, SPACE> >(*tangle, *tangle, 0.2f);
    gettimeofday(&begin, 0);
    (*isosurface)();
    std::cout << "Number of vertices: " << isosurface->numTotalVertices << std::endl;
#endif

#ifdef CUTPLANE
    tangle = new tangle_field<int, float, SPACE>(grid_size, grid_size, grid_size);
    plane = new plane_field<int, float, SPACE>(make_float3(0.0f, 0.0f, grid_size/2), make_float3(0.0f, 0.0f, 1.0f), grid_size, grid_size, grid_size);
    cutplane = new marching_cube<plane_field<int, float, SPACE>, tangle_field<int, float, SPACE> >(*plane, *tangle, 0.2f);
    gettimeofday(&begin, 0);
    (*cutplane)();
    std::cout << "Number of vertices: " << cutplane->numTotalVertices << std::endl;
#endif

#ifdef THRESHOLD
    scalar_field = new sphere_field<int, float, SPACE>(grid_size, grid_size, grid_size);
    threshold = new threshold_geometry<sphere_field<int, float, SPACE> >(*scalar_field, 4, 1600);
    gettimeofday(&begin, 0);
    (*threshold)();
    std::cout << "Number of vertices: " << threshold->numTotalVertices << std::endl;
#endif

    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);
    frame_count++;
    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    char title[256];
    sprintf(title, "Marching Cube, fps: %2.2f", float(frame_count)/seconds);
    std::cout << title << std::endl;
    seconds = 0.0f;

    return 0;
}

