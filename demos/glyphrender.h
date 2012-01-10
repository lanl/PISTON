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

#ifndef GlyphRender_H
#define GlyphRender_H

#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>

#include <GL/glut.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <vtkSphereSource.h>
#include <vtkArrowSource.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>

#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataNormals.h>

#include <sys/time.h>
#include "quaternion.h"

#include <cutil_math.h>
#include <piston/choose_container.h>

using namespace piston;
#define SPACE thrust::detail::default_device_space_tag

template <typename ValueType>
struct color_map : thrust::unary_function<ValueType, float4>
{
    const ValueType min;
    const ValueType max;
    const ValueType minOffset;
    const ValueType maxOffset;
    const bool reversed;

    __host__ __device__
    color_map(ValueType min, ValueType max, bool reversed=false) : min(min), max(max), reversed(reversed), minOffset(min-0.0*(max-min)), maxOffset(max+0.0*(max-min)) {}

    __host__ __device__
    float4 operator()(ValueType val)
    {
	// HSV rainbow for height field, stolen form Manta
    if (val < -1000.0) return make_float4(0.0, 0.0, 0.0, 1.0);
    if (val > max) val = maxOffset; //return make_float4(1.0, 1.0, 1.0, 1.0);
    if (val < min) val = minOffset; //return make_float4(0.0, 0.0, 0.0, 0.0);
	const float V = 0.7f, S = 1.0f;
	float H;
	if (reversed) H = (static_cast<float> (val - minOffset) / (maxOffset - minOffset));
	else H = (1.0f - static_cast<float> (val - minOffset) / (maxOffset - minOffset));

	if (H < 0.0f) H = 0.0f;
	else if (H > 1.0f) H = 1.0f;
	H *= 4.0f;

	float i = floor(H);
	float f = H - i;

	float p = V * (1.0 - S);
	float q = V * (1.0 - S * f);
	float t = V * (1.0 - S * (1 - f));

	float R, G, B;
	if (i == 0.0) { R = V; G = t; B = p; }
        else if (i == 1.0) { R = q; G = V; B = p; }
        else if (i == 2.0) { R = p; G = V; B = t; }
        else if (i == 3.0) { R = p; G = q; B = V; }
        else if (i == 4.0) { R = t; G = p; B = V; }
        else { R = V; G = p; B = q; }

	return make_float4(R, G, B, 1.0);
    }
};

#include <piston/image3d.h>
#include <piston/vtk_image3d.h>
#include <piston/glyph.h>

class GlyphRender
{
public:
  GlyphRender();
  void setZoomLevelPct(float pct);
  void display();
  void idle();
  void initGL(bool aAllowInterop);
  void cleanup();
  int read();
  void resetView();
  void copyPolyData(vtkPolyData *polyData, thrust::device_vector<float3> &points, thrust::device_vector<float3> &vectors, thrust::device_vector<uint3> &indices);

  bool useInterop;
  float3 center_pos;
  float3 camera_up;
  float cameraZ, cameraFOV, zoomLevelBase;
  int mouse_old_x, mouse_old_y;
  Quaternion qrot;
  bool includeInput, includeGlyphs;

  int mouse_buttons;
  float3 translate;
  float rotationMatrix[16];
  Quaternion qDefault;
  int viewportWidth, viewportHeight;

  float maxValue, minValue;

  float zoomLevelPct, zoomLevelPctDefault;

  GLuint vboBuffers[4];  struct cudaGraphicsResource* vboResources[4];

  thrust::host_vector<float3>  vertices;
  thrust::host_vector<float3>  normals;
  thrust::host_vector<float4>  colors;
  thrust::host_vector<uint3> indices;

  thrust::host_vector<float3>  inputVerticesHost;
  thrust::host_vector<float3>  inputNormalsHost;
  thrust::host_vector<float4>  inputColorsHost;
  thrust::host_vector<uint3> inputIndicesHost;

  thrust::device_vector<float3>  inputVertices;
  thrust::device_vector<float3>  inputNormals;
  thrust::device_vector<float>   inputScalars;
  thrust::device_vector<float4>  inputColors;
  thrust::device_vector<uint3> inputIndices;
  thrust::device_vector<float3>  glyphVertices;
  thrust::device_vector<float3>  glyphNormals;
  thrust::device_vector<float4>  glyphColors;
  thrust::device_vector<uint3> glyphIndices;

  vtkArrowSource *arrowSource;
  vtkSphereSource *sphereSource;
  vtkPolyData *spherePoly;
  vtkPolyData *arrowPoly;
  vtkTriangleFilter *triangleFilter;
  vtkPolyDataNormals *normalGenerator;

  glyph<thrust::device_vector<float3>::iterator, thrust::device_vector<float3>::iterator, thrust::device_vector<float>::iterator,
        thrust::device_vector<float3>::iterator, thrust::device_vector<float3>::iterator, thrust::device_vector<uint3>::iterator >* glyphs;

};

#endif
