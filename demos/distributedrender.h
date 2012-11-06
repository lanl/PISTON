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

#ifndef DISTRIBUTEDRENDER_H
#define DISTRIBUTEDRENDER_H

#ifdef __APPLE__
    #include <GL/glew.h>
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/glew.h>
    #include <GL/glut.h>
    #include <GL/gl.h>
#endif


#include "piston/util/quaternion.h"
#include <piston/piston_math.h>
#include <piston/choose_container.h>

#include <piston/util/tangle_field.h>
#include <piston/marching_cube.h>
#include <piston/dmarching_cube.h>

using namespace piston;
#define SPACE thrust::detail::default_device_space_tag
#define GRID_SIZE 16



class DistributedRender
{
public:
  DistributedRender();
  void setIsovaluePct(float pct);
  void setZoomLevelPct(float pct);
  void display();
  void idle();
  void initContour();
  void initGL();
  void timeContours();
  void cleanup();
  void resetView();

  float3 center_pos;
  float3 camera_up;
  float cameraZ, cameraFOV, zoomLevelBase, zNear, zFar;
  float maxIso, minIso, isoIter, deltaIso; 
  float isovalue;
  Quaternion qrot;
  float rotationMatrix[16];

  float zoomLevelPct, zoomLevelPctDefault;
  float isovaluePct;

  thrust::host_vector<float4> vertices;
  thrust::host_vector<float4> vertices2;
  thrust::host_vector<float3> normals;
  thrust::host_vector<float3> normals2;
  thrust::host_vector<float4> colors;

  tangle_field</*int, float,*/ SPACE>* cayley;
  tangle_field</*int, float,*/ SPACE>* cayley2;
  marching_cube<tangle_field</*int, float,*/ SPACE>, tangle_field</*int, float,*/ SPACE> >* contour2;
  dmarching_cube<tangle_field</*int, float,*/ SPACE>, tangle_field</*int, float,*/ SPACE> >* contour;

};

#endif
