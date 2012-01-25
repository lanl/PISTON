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

#ifndef ISORENDER_H
#define ISORENDER_H

#ifdef __APPLE__
    #include <GL/glew.h>
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/glew.h>
    #include <GL/glut.h>
    #include <GL/gl.h>
#endif

#include <cuda_gl_interop.h>
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

#include <cutil_math.h>
#include <piston/choose_container.h>

using namespace piston;
#define SPACE thrust::detail::default_device_space_tag

#define BIG_CONTOUR_BUFFER_SIZE 25000000
#define BIG_PLANE_BUFFER_SIZE 13000000

#define CONTOUR_BUFFER_SIZE 12000000
#define PLANE_BUFFER_SIZE 2100000
#define CONSTANT_BUFFER_SIZE 5000000

typedef enum {
	DEFAULT_MODE = 0,
	ISOSURFACE_MODE,
	CUT_SURFACE_MODE,
	THRESHOLD_MODE
} UserModes;


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
  void createBuffers();
  void createOperators();
  int read(char* aFileName, int aMode);
  int read(int aDataSetIndex, int aNumIters=0, int aSaveFrames=0, char* aFrameDirectory=0);
  void resetView();

  int userMode;
  char userFileName[2048];
  bool userRange;
  float userMin, userMax;
  int userBufferSize;

  static const int numDataSets = 9;
  int xMin, yMin, zMin, xMax, yMax, zMax;

  bool useInterop;
  int NPoints;
  int numIters;
  float3 center_pos;
  float3 camera_up;
  float cameraZ, cameraFOV, zoomLevelBase, zNear, zFar;
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
  float3 plane_normal;

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
  float planeMax;

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
