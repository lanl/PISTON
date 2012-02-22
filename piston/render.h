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

#ifndef RENDER_H_
#define RENDER_H_

#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <piston/image3d.h>
#include <piston/piston_math.h>
#include <piston/choose_container.h>


namespace piston {

template <typename InputVertices>
class render
{
public:
    InputVertices inputVertices;
    typedef typename thrust::counting_iterator<int> CountingIterator;
    int nVertices, width, height, pixelSize;
    thrust::device_vector<char> frame;

    render(InputVertices inputVertices, int nVertices, int width, int height) : inputVertices(inputVertices), nVertices(nVertices), width(width), height(height), pixelSize(4)
    {
      frame.resize(width*height*pixelSize);
    };

    void operator()()
    {
      thrust::fill(frame.begin(), frame.end(), 255);
      thrust::for_each(CountingIterator(0), CountingIterator(0)+nVertices/3, scanline(inputVertices, nVertices, width, height, pixelSize,
                                                                                      thrust::raw_pointer_cast(&*frame.begin())));
    }

    struct scanline : public thrust::unary_function<int, void>
    {
      InputVertices inputVertices;
      char* frame;
      int nVertices, width, height, pixelSize;

      __host__ __device__
      scanline(InputVertices inputVertices, int nVertices, int width, int height, int pixelSize, char* frame) :
               inputVertices(inputVertices), nVertices(nVertices), width(width), height(height), pixelSize(pixelSize), frame(frame) {};

      __host__ __device__
      void operator() (int id) const
      {
        int minY = height;  int maxY = 0;
        for (unsigned int v=0; v<3; v++)
        {
          float3 vertex = *(inputVertices + 3*id + v);
          int i = (int)(vertex.x);   int j = (int)(vertex.y);
          frame[i*height*pixelSize + j*pixelSize + 0] = 0;
          frame[i*height*pixelSize + j*pixelSize + 1] = 0;
          frame[i*height*pixelSize + j*pixelSize + 2] = 255;
          //frame[i*height*pixelSize + j*pixelSize + 3] = 255;
          if (j < minY) minY = j;  if (j > maxY) maxY = j;
        }

        //std::cout << "Range " << minY << " " << maxY << std::endl;

        for (unsigned int s=minY; s<=maxY; s++)
        {
          float3 v0 = *(inputVertices + 3*id + 0);
          float3 v1 = *(inputVertices + 3*id + 1);
          float3 v2 = *(inputVertices + 3*id + 2);

          int x0 = -99999;
          if (((s > v0.y) && (s < v1.y)) || ((s < v0.y) && (s > v1.y)))
          {
            float m0 = (v0.y - v1.y) / (v0.x - v1.x);
            float b0 = v0.y - m0*v0.x;
            x0 = (int)((s - b0)/m0);
          }

          int x1 = -99999;
          if (((s > v2.y) && (s < v1.y)) || ((s < v2.y) && (s > v1.y)))
          {
            float m1 = (v2.y - v1.y) / (v2.x - v1.x);
            float b1 = v2.y - m1*v2.x;
            x1 = (int)((s - b1)/m1);
          }

          int x2 = -99999;
          if (((s > v0.y) && (s < v2.y)) || ((s < v0.y) && (s > v2.y)))
          {
            float m2 = (v0.y - v2.y) / (v0.x - v2.x);
            float b2 = v0.y - m2*v0.x;
            x2 = (int)((s - b2)/m2);
          }

          int sb = 99999;
          if ((x0 > -9999) && (x0 < sb)) sb = x0;
          if ((x1 > -9999) && (x1 < sb)) sb = x1;
          if ((x2 > -9999) && (x2 < sb)) sb = x2;

          int ss = -99999;
          if ((x0 > -9999) && (x0 > ss)) ss = x0;
          if ((x1 > -9999) && (x1 > ss)) ss = x1;
          if ((x2 > -9999) && (x2 > ss)) ss = x2;

          //std::cout << "X: " << sb << " " << ss << std::endl;

          if ((sb >= 0) && (sb < width) && (ss >= 0) && (ss < width))
          {
            for (unsigned int f=sb; f<=ss; f++)
            {
              frame[f*height*pixelSize + s*pixelSize + 0] = 255;
              frame[f*height*pixelSize + s*pixelSize + 1] = 0;
              frame[f*height*pixelSize + s*pixelSize + 2] = 0;
            }
          }
          //std::cout << "Intersections " << x0 << " " << x1 << " " << x2 << std::endl;

        }
      }
    };

    thrust::device_vector<char>::iterator frame_begin() { return frame.begin(); }
    thrust::device_vector<char>::iterator frame_end() { return frame.end(); }

};

}

#endif /* RENDER_H_ */
