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

#ifndef TetraRender_H
#define TetraRender_H

#include <GL/glew.h>
#include <GL/gl.h>
#ifdef USE_INTEROP
#include <cuda_gl_interop.h>
#endif

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
#include <vtkDataSetTriangleFilter.h>
#include <vtkDataSetToUnstructuredGridFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLMultiBlockDataReader.h>

#include <sys/time.h>


#include <vtkImageData.h>
#include <vtkRTAnalyticSource.h>

#include <piston/vtk_image3d.h>
#include <piston/piston_math.h>
#include <piston/hsv_color_map.h>
#include "piston/image3d_to_tetrahedrons.h"
#include "piston/marching_tetrahedron.h"


#include "piston/util/quaternion.h"


#include <piston/choose_container.h>


using namespace piston;
#define SPACE thrust::detail::default_device_space_tag

#include <piston/image3d.h>
#include <piston/vtk_image3d.h>
#include "piston/util/height_field.h"
#include <piston/image3d_to_tetrahedrons.h>
#include <piston/unstructured_tetrahedra.h>
#include <piston/marching_tetrahedron.h>

class TetraRender
{
public:
  TetraRender(char* a_filename, bool a_computeAverageIsovalue, float a_isovalue=0.0f);
  void setZoomLevelPct(float pct);
  void display();
  void idle();
  void initGL(bool aAllowInterop);
  void cleanup();
  void resetView();

  bool useInterop;
  float cameraZ, cameraFOV, zoomLevelBase;
  int mouse_old_x, mouse_old_y;
  Quaternion qrot;

  int mouse_buttons;
  float3 translate;
  float rotationMatrix[16];
  Quaternion qDefault;
  int viewportWidth, viewportHeight;

  float zoomLevelPct, zoomLevelPctDefault;

  GLuint vboBuffers[4];  struct cudaGraphicsResource* vboResources[4];

  thrust::host_vector<float4>  vertices;
  thrust::host_vector<float3>  normals;
  thrust::host_vector<float4>  colors;
  thrust::host_vector<float>  scalars;

  bool showIso;
  float3 centerPos, lookPos, cameraUp;
  float isovalue;
  bool computeAverageIsovalue;
  char filename[512];
  float minValue, maxValue;
  int gridSize;
  int wireMode;

  vtkRTAnalyticSource* src;
  vtkXMLUnstructuredGridReader* reader;
  vtkUnstructuredGrid* tetrahedra;
  unstructured_tetrahedra<SPACE>* utet;
  vtkDataSetTriangleFilter* triFilter;
  marching_tetrahedron<unstructured_tetrahedra<SPACE>, unstructured_tetrahedra<SPACE> >* isosurface;
};

#endif
