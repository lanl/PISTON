/*
Copyright (c) 2011, Los Alamos National Security, LLC
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
    	and/or other materials provided with the distribution.
    Neither the name of the Los Alamos National Laboratory nor the names of its contributors may be used to endorse or promote products derived from this
    	software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef ISORENDER_H
#define ISORENDER_H

#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>

#include <GL/glut.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include "vtkImageReader.h"
#include "vtkImageData.h"
#include "vtkPolyData.h"
#include "vtkXMLImageDataReader.h"
#include "vtkRectilinearGridReader.h"
#include "vtkPLYReader.h"
#include "vtkCellArray.h"
#include <sys/time.h>
#include "quaternion.h"

#include <piston/cutil_math.h>
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
    color_map(ValueType min, ValueType max, bool reversed=false) : min(min), max(max), reversed(reversed), minOffset(min-0.1*(max-min)), maxOffset(max+0.1*(max-min)) {}

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
#include <piston/util/plane_field.h>
#include <piston/marching_cube.h>
#include <piston/threshold_geometry.h>


class IsoRender
{
public:
  IsoRender();
  void setIsovaluePct(float pct);
  void setPlaneLevelPct(float pct);
  void setZoomLevelPct(float pct);
  void display();
  void idle();
  void initGL(bool aAllowInterop, bool aBigDemo, bool aShowLabels, int aDataSet);
  void timeContours();
  void cleanup();
  void screenShot(std::string fileName, unsigned int width, unsigned int height, bool includeAlpha = false );
  int read(int aDataSetIndex, int aNumIters=0, int aSaveFrames=0, char* aFrameDirectory=0);
  void resetView();

  static const int numDataSets = 9;
  int xMin, yMin, zMin, xMax, yMax, zMax;

  bool useInterop;
  int NPoints;
  int numIters;
  float3 center_pos;
  float3 camera_up;
  float cameraZ, cameraFOV, zoomLevelBase;
  float maxIso, minIso, isoIter, deltaIso; 
  float minThreshold, maxThreshold, thresholdFloor, threshold;
  float isovalue, delta, planeLevel;
  int mouse_old_x, mouse_old_y;
  int saveFrames;
  char frameDirectory[1024];
  double timerTotal;
  int timerCount;
  int ncells;
  double polyAvgX, polyAvgY, polyAvgZ;
  float3 polyOffset;
  Quaternion qrot;

  bool animate;
  bool showLabels;
  int mouse_buttons;
  float3 translate;
  float rotationMatrix[16];
  Quaternion qDefault;
  int frameCount;
  float lastIsovalue, lastPlaneLevel, lastThreshold;
  int includePlane;
  int includeContours, useContours;
  int includeThreshold, useThreshold;
  int includeConstantContours, useConstantContours;
  int includePolygons;
  int discardMinVals;
  int bigDemo;
  int viewportWidth, viewportHeight;

  float zoomLevelPct, zoomLevelPctDefault;
  float planeLevelPct;
  float isovaluePct;
  float thresholdPct;
  int dataSetIndex;
  float polyScale;
  float curFPS;

  GLuint vboBuffers[3];  struct cudaGraphicsResource* vboResources[3];
  GLuint planeBuffers[3];  struct cudaGraphicsResource* planeResources[3];
  GLuint constantBuffers[3];  struct cudaGraphicsResource* constantResources[3];

  thrust::host_vector<float4> vertices;
  thrust::host_vector<float3> normals;
  thrust::host_vector<float4> colors;

  thrust::host_vector<float4> planeVertices;
  thrust::host_vector<float3> planeNormals;
  thrust::host_vector<float4> planeColors;

  thrust::host_vector<float4> constantVertices;
  thrust::host_vector<float3> constantNormals;
  int numConstantVertices;

  std::vector<vtk_image3d<int, float, SPACE>*> images;
  std::vector<plane_field<int, float, SPACE>*> planeFields;
  std::vector<marching_cube<vtk_image3d<int, float, SPACE>, vtk_image3d<int, float, SPACE> >*> contours;
  std::vector<marching_cube<plane_field<int, float, SPACE>, vtk_image3d<int, float, SPACE> >*> planeContours;
  std::vector<marching_cube<vtk_image3d<int, float, SPACE>, vtk_image3d<int, float, SPACE> >*> constantContours;
  std::vector<threshold_geometry<vtk_image3d<int, float, SPACE> >*> thresholds;

  vtkImageData *output;
  vtkIdType npts;

  std::vector<vtkXMLImageDataReader*> readers;
  std::vector<vtkPLYReader*> plyReaders;
  std::vector<vtkPolyData*> polyData;
  vtkCellArray* polyTriangles;
  vtkIdType *curTriangle;
};

#endif
