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

#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <float.h>

#include "glyphrender.h"

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

#define GLYPH_BUFFER_SIZE 12000000


GlyphRender::GlyphRender()
{
    mouse_buttons = 0;
    translate = make_float3(0.0, 0.0, 0.0);
}


void GlyphRender::setZoomLevelPct(float pct)
{
    if (pct > 1.0) pct = 1.0;  if (pct < 0.0) pct = 0.0;
    zoomLevelPct = pct;
    cameraFOV = 0.0 + zoomLevelBase*pct;
}


void GlyphRender::resetView()
{
    qrot.set(qDefault.x, qDefault.y, qDefault.z, qDefault.w);
    zoomLevelPct = zoomLevelPctDefault;
    cameraFOV = 0.0 + zoomLevelBase*zoomLevelPct;
}


void GlyphRender::display()
{
    if (true)
    {
      if (useInterop)
      {
        for (int i=0; i<4; i++) glyphs->vboResources[i] = vboResources[i];
    	glyphs->minValue = minValue;  glyphs->maxValue = maxValue;
      }

      (*(glyphs))();

      if (!useInterop)
      {
    	normals.assign(glyphs->normals_begin(), glyphs->normals_end());
    	indices.assign(glyphs->indices_begin(), glyphs->indices_end());
    	vertices.assign(glyphs->vertices_begin(), glyphs->vertices_end());
    	colors.assign(thrust::make_transform_iterator(glyphs->scalars_begin(), color_map<float>(minValue, maxValue)),
    	              thrust::make_transform_iterator(glyphs->scalars_end(), color_map<float>(minValue, maxValue)));
      }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(cameraFOV, 2.0, 0.01, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    /*gluLookAt(center_pos.x, center_pos.y, cameraZ,
                  center_pos.x, center_pos.y, center_pos.z,
                  camera_up.x, camera_up.y, camera_up.z);*/
    gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0);
    glPushMatrix();

    //float3 center = make_float3(center_pos.x, center_pos.y, center_pos.z);

    qrot.getRotMat(rotationMatrix);
    glMultMatrixf(rotationMatrix);

    /*GLfloat matrix[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
    float3 offset = make_float3(matrix[0]*center.x + matrix[1]*center.y + matrix[2]*center.z, matrix[4]*center.x + matrix[5]*center.y + matrix[6]*center.z,
                                matrix[8]*center.x + matrix[9]*center.y + matrix[10]*center.z);
    offset.x = center.x - offset.x; offset.y = center.y - offset.y; offset.z = center.z - offset.z;
    glTranslatef(-offset.x, -offset.y, -offset.z);*/

    glColor4f(0.5, 0.5, 0.5, 1.0);

    if (includeInput)
    {
      glColor4f(0.5, 0.5, 0.5, 1.0);
      glEnableClientState(GL_NORMAL_ARRAY);
      glDisableClientState(GL_COLOR_ARRAY);
      glEnableClientState(GL_VERTEX_ARRAY);

      glNormalPointer(GL_FLOAT, 0, &inputNormalsHost[0]);
      //glColorPointer(4, GL_FLOAT, 0, &inputColorsHost[0]);
      glVertexPointer(3, GL_FLOAT, 0, &inputVerticesHost[0]);
      glDrawElements(GL_TRIANGLES, 3*inputIndicesHost.size(), GL_UNSIGNED_INT, &inputIndicesHost[0]);
    }

    if (includeGlyphs)
    {
      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_COLOR_ARRAY);
      glEnableClientState(GL_NORMAL_ARRAY);

      if (useInterop)
      {
        glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[0]);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboBuffers[3]);
        glVertexPointer(3, GL_FLOAT, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[1]);
        glColorPointer(4, GL_FLOAT, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[2]);
        glNormalPointer(GL_FLOAT, 0, 0);
        glDrawElements(GL_TRIANGLES, 3*glyphs->numIndices, GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
      }
      else
      {
        glNormalPointer(GL_FLOAT, 0, &normals[0]);
        glColorPointer(4, GL_FLOAT, 0, &colors[0]);
        glVertexPointer(3, GL_FLOAT, 0, &vertices[0]);
        glDrawElements(GL_TRIANGLES, 3*indices.size(), GL_UNSIGNED_INT, &indices[0]);
      }
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glPopMatrix();
}


void GlyphRender::cleanup()
{
	if (useInterop)
	{
	  printf("Deleting VBO\n");
	  if (vboBuffers[0])
	  {
	    for (int i=0; i<4; i++) cudaGraphicsUnregisterResource(vboResources[i]);
	    for (int i=0; i<4; i++)
	    {
	      glBindBuffer(1, vboBuffers[i]);
	      glDeleteBuffers(1, &(vboBuffers[i]));
	      vboBuffers[i] = 0;
	    }
	  }
	}
	else
	{
	  vertices.clear(); normals.clear(); colors.clear();
	}
}


void GlyphRender::initGL(bool aAllowInterop)
{
#ifdef USE_INTEROP
    useInterop = aAllowInterop;
#else
    useInterop = false;
#endif

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    float white[] = { 0.5, 0.5, 0.5, 1.0 };
    float black[] = { 0.0, 0.0, 0.0, 1.0 };
    float lightPos[] = { 0.0, 0.0, 10.0, 1.0 };
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100);
    glLightfv(GL_LIGHT0, GL_AMBIENT, white);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, black);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);

    glMatrixMode(GL_PROJECTION);
    gluPerspective(cameraFOV, 2.0, 200.0, 4000.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(center_pos.x, center_pos.y, cameraZ,
              center_pos.x, center_pos.y, center_pos.z,
              camera_up.x, camera_up.y, camera_up.z);

    if (useInterop)
    {
      glewInit();
      cudaGLSetGLDevice(0);

      // initialize contour buffer objects
      glGenBuffers(4, vboBuffers);
      for (int i=0; i<3; i++)
      {
        unsigned int buffer_size = (i == 2) ? GLYPH_BUFFER_SIZE*sizeof(float3) : GLYPH_BUFFER_SIZE*sizeof(float4);
        glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[i]);
        glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
      }
      glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[3]);
      glBufferData(GL_ARRAY_BUFFER, GLYPH_BUFFER_SIZE*sizeof(uint3), 0, GL_DYNAMIC_DRAW);

      glBindBuffer(GL_ARRAY_BUFFER, 0);
      for (int i=0; i<4; i++) cudaGraphicsGLRegisterBuffer(&(vboResources[i]), vboBuffers[i], cudaGraphicsMapFlagsWriteDiscard);
    }

    //printf("Error code: %s\n", cudaGetErrorString(errorCode));
    read();
}


void GlyphRender::copyPolyData(vtkPolyData *polyData, thrust::device_vector<float3> &points, thrust::device_vector<float3> &vectors, thrust::device_vector<uint3> &indexes)
{
	vtkPoints* pts = polyData->GetPoints();
	vtkFloatArray* verts = vtkFloatArray::SafeDownCast(pts->GetData());
	vtkFloatArray* norms = vtkFloatArray::SafeDownCast(polyData->GetPointData()->GetNormals());
    float3* vData = (float3*)verts->GetPointer(0);
	float3* nData = (float3*)norms->GetPointer(0);
	points.assign(vData, vData+verts->GetNumberOfTuples());
	vectors.assign(nData, nData+norms->GetNumberOfTuples());

	vtkCellArray* cellArray = polyData->GetPolys();
	vtkIdTypeArray* conn = cellArray->GetData();
	vtkIdType* cData = conn->GetPointer(0);
	for (int i=0; i<3*polyData->GetNumberOfPolys(); i++) cData[i] = cData[(i/3)*4+(i%3)+1];
	thrust::host_vector<uint> indexTemp;
	indexTemp.assign(cData, cData+3*polyData->GetNumberOfPolys());
	uint3* c3Data = (uint3*)(thrust::raw_pointer_cast(&*indexTemp.begin()));
	indexes.assign(c3Data, c3Data+polyData->GetNumberOfPolys());
}


int GlyphRender::read()
{
	sphereSource = vtkSphereSource::New();
	sphereSource->Update();
	spherePoly = vtkPolyData::New();
	spherePoly->ShallowCopy(sphereSource->GetOutput());
	copyPolyData(spherePoly, inputVertices, inputNormals, inputIndices);

    arrowSource = vtkArrowSource::New();
    arrowSource->Update();
    triangleFilter = vtkTriangleFilter::New();
    triangleFilter->SetInputConnection(arrowSource->GetOutputPort());
    triangleFilter->Update();
	arrowPoly = vtkPolyData::New();
	arrowPoly->ShallowCopy(triangleFilter->GetOutput());

	normalGenerator = vtkPolyDataNormals::New();
	normalGenerator->SetInput(arrowPoly);
	normalGenerator->ComputePointNormalsOn();
	normalGenerator->ComputeCellNormalsOff();
	normalGenerator->Update();
	arrowPoly = normalGenerator->GetOutput();

	copyPolyData(arrowPoly, glyphVertices, glyphNormals, glyphIndices);

	inputScalars.resize(inputVertices.size());
	maxValue = -FLT_MAX;  minValue = FLT_MAX;
	for (int i=0; i<inputVertices.size(); i++)
	{
		float3 curVertex = inputVertices[i];
		if (curVertex.x > maxValue) maxValue = curVertex.x;
		if (curVertex.x < minValue) minValue = curVertex.x;
	}
	for (int i=0; i<inputScalars.size(); i++)
	{
		float3 curVertex = inputVertices[i];
		inputScalars[i] = 0.2 + 0.8*((curVertex.x - minValue) / (maxValue - minValue)); //0.2 + 1.6*0.01*(rand() % 100);
	}
	minValue = 0.2;  maxValue = 1.0;

	glyphs = new glyph<thrust::device_vector<float3>::iterator, thrust::device_vector<float3>::iterator, thrust::device_vector<float>::iterator,
			           thrust::device_vector<float3>::iterator, thrust::device_vector<float3>::iterator, thrust::device_vector<uint3>::iterator>
                      (inputVertices.begin(), inputNormals.begin(), inputScalars.begin(), glyphVertices.begin(), glyphNormals.begin(), glyphIndices.begin(),
                       inputVertices.size(), glyphVertices.size(), glyphIndices.size());

	glyphs->useInterop = useInterop;
	zoomLevelBase = cameraFOV = 40.0; cameraZ = 2.0; zoomLevelPct = zoomLevelPctDefault = 0.5;
	center_pos = make_float3(0, 0, 0);
	cameraFOV = zoomLevelBase*zoomLevelPct;  camera_up = make_float3(0,1,0);

	includeGlyphs = true; includeInput = true;
	inputIndicesHost = inputIndices;
	inputNormalsHost = inputNormals;
	inputVerticesHost = inputVertices;

    return 0;
}
