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

#include "tetrarender.h"

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

#define TETRA_BUFFER_SIZE 12000000
#define BASE_SCALE 3.5


TetraRender::TetraRender(char* a_filename, bool a_computeAverageIsovalue, float a_isovalue)
{
    strcpy(filename, a_filename);
    computeAverageIsovalue = a_computeAverageIsovalue;
    isovalue = a_isovalue;
    mouse_buttons = 0;
    translate = make_float3(0.0, 0.0, 0.0);
    wireMode = 0;
}


void TetraRender::setZoomLevelPct(float pct)
{
    if (pct > 1.0) pct = 1.0;  if (pct < 0.0) pct = 0.0;
    zoomLevelPct = pct;
    cameraFOV = 0.0 + zoomLevelBase*pct;
}


void TetraRender::resetView()
{
    qrot.set(qDefault.x, qDefault.y, qDefault.z, qDefault.w);
    zoomLevelPct = zoomLevelPctDefault;
    cameraFOV = 0.0 + zoomLevelBase*zoomLevelPct;
}


void TetraRender::display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int minPass, maxPass;  minPass = maxPass = 0;
    if (wireMode == 0) { minPass = 0; maxPass = 1; }
    if (wireMode == 1) { minPass = 1; maxPass = 2; }
    if (wireMode == 2) { minPass = 0; maxPass = 2; }

isovalue *= 1.001;

    for (unsigned int pm=minPass; pm<maxPass; pm++)
    {
      isosurface->set_isovalue(isovalue);
      ((*isosurface)());

      if (!useInterop)
      {
        normals.assign(isosurface->normals_begin(), isosurface->normals_end());
        vertices.assign(isosurface->vertices_begin(), isosurface->vertices_end());
        colors.assign(thrust::make_transform_iterator(isosurface->scalars_begin(), color_map<float>(minValue, maxValue)),
    	            thrust::make_transform_iterator(isosurface->scalars_end(), color_map<float>(minValue, maxValue)));
      }
    
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);      
      glDisable(GL_POLYGON_OFFSET_LINE);
      if (pm == 1)
      {
        glEnable(GL_POLYGON_OFFSET_LINE);
        glPolygonOffset(1.0f,1.0f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
      } 

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(cameraFOV, 2.0f, BASE_SCALE/3.5f, 5.0f*BASE_SCALE); 

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      gluLookAt(centerPos.x, -2.0f*BASE_SCALE, centerPos.z, centerPos.x, 0, centerPos.z, cameraUp.x, cameraUp.y, cameraUp.z); 
      glPushMatrix();

      glTranslatef(lookPos.x, lookPos.y, lookPos.z);
      if (pm == 1) glTranslatef(0.0f, -BASE_SCALE*0.03f, 0.0f);
      qrot.getRotMat(rotationMatrix);
      float3 offset = matrixMul(rotationMatrix, centerPos);

      glMultMatrixf(rotationMatrix);
      glTranslatef(offset.x-centerPos.x, offset.y-centerPos.y, offset.z-centerPos.z);

      glEnableClientState(GL_VERTEX_ARRAY);
      if (pm == 0) glEnableClientState(GL_COLOR_ARRAY);
      glEnableClientState(GL_NORMAL_ARRAY);

      #ifdef USE_INTEROP
        if (useInterop)
        {
          glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[0]);
          glVertexPointer(4, GL_FLOAT, 0, 0);
          glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[1]);
          glColorPointer(4, GL_FLOAT, 0, 0);
          glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[2]);
          glNormalPointer(GL_FLOAT, 0, 0);
          glDrawArrays(GL_TRIANGLES, 0, isosurface->num_total_vertices);
          glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
        else
      #endif
      {
        if (showIso)
        {
          glNormalPointer(GL_FLOAT, 0, &normals[0]);
          glColorPointer(4, GL_FLOAT, 0, &colors[0]);
          glVertexPointer(4, GL_FLOAT, 0, &vertices[0]);
          glDrawArrays(GL_TRIANGLES, 0, vertices.size());
        }
      }

      glDisableClientState(GL_VERTEX_ARRAY);
      glDisableClientState(GL_COLOR_ARRAY);
      glDisableClientState(GL_NORMAL_ARRAY);

      glPopMatrix();
    }
}


void TetraRender::cleanup()
{
    #ifdef USE_INTEROP
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
    #endif
    {
      vertices.clear(); normals.clear(); colors.clear();
    }
}


void TetraRender::initGL(bool aAllowInterop)
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

    #ifdef USE_INTEROP
      if (useInterop)
      {
        glewInit();
        cudaGLSetGLDevice(0);

        glGenBuffers(4, vboBuffers);
        for (int i=0; i<3; i++)
        {
          unsigned int buffer_size = (i == 2) ? TETRA_BUFFER_SIZE*sizeof(float3) : TETRA_BUFFER_SIZE*sizeof(float4);
          glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[i]);
          glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
        }
        glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[3]);
        glBufferData(GL_ARRAY_BUFFER, TETRA_BUFFER_SIZE*sizeof(uint3), 0, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        for (int i=0; i<4; i++) cudaGraphicsGLRegisterBuffer(&(vboResources[i]), vboBuffers[i], cudaGraphicsMapFlagsWriteDiscard);
      }
    #endif

    triFilter = vtkDataSetTriangleFilter::New();
    triFilter->TetrahedraOnlyOn();

    char fullFilename[1024];
    strcpy(fullFilename, filename);
    reader = vtkXMLUnstructuredGridReader::New();
    int fileFound = reader->CanReadFile(fullFilename);
    if (!fileFound)
    {
      sprintf(fullFilename, "%s/%s", STRINGIZE_VALUE_OF(DATA_DIRECTORY), filename);
      fileFound = reader->CanReadFile(fullFilename);
    }
    if (fileFound)
    {
      reader->SetFileName(fullFilename);  
      reader->Update();
      triFilter->SetInput(reader->GetOutput());
    } 
    else 
    {
      gridSize = atoi(filename);
      src = vtkRTAnalyticSource::New();
      src->SetWholeExtent(-gridSize, gridSize, -gridSize, gridSize, -gridSize, gridSize);
      src->Update();
      triFilter->SetInput(src->GetOutput());
    }

    triFilter->Update();
    tetrahedra = triFilter->GetOutput();

    double curVertex[3], minVertex[3], maxVertex[3];
    for (unsigned int i=0; i<3; i++) { minVertex[i] = DBL_MAX; maxVertex[i] = -DBL_MAX; }
    for (unsigned int i=0; i<tetrahedra->GetNumberOfPoints(); i++)
    {
      tetrahedra->GetPoint(i, curVertex);
      for (unsigned int j=0; j<3; j++)
      {
        if (minVertex[j] > curVertex[j]) minVertex[j] = curVertex[j];
        if (maxVertex[j] < curVertex[j]) maxVertex[j] = curVertex[j];
      }
    }
    double maxRange = 0.0;
    for (unsigned int i=0; i<3; i++) 
    { 
      double curRange = maxVertex[i] - minVertex[i];
      if (curRange > maxRange) maxRange = curRange; 
    }
    std::cout << "Range: " << maxRange << std::endl;
        
    utet = new unstructured_tetrahedra<SPACE>(tetrahedra, BASE_SCALE/maxRange);
    
    if (computeAverageIsovalue)
    {
      vtkDataArray* array1 = tetrahedra->GetPointData()->GetArray(0);
      vtkFloatArray* farray = vtkFloatArray::SafeDownCast(array1);
      float* rawData = farray->GetPointer(0);
      isovalue = 0.0f;
      for (unsigned int i=0; i<array1->GetNumberOfTuples(); i++) isovalue += rawData[i];
      isovalue /= (1.0f*array1->GetNumberOfTuples());
      std::cout << "Isovalue: " << isovalue << std::endl;
    }
    minValue = isovalue*0.9999f;  maxValue = isovalue;
 
    showIso = true;  
    zoomLevelBase = cameraFOV = 60.0f; cameraZ = 2.0f; zoomLevelPct = zoomLevelPctDefault = 0.5f;
    cameraFOV = zoomLevelBase*zoomLevelPct;  cameraUp = make_float3(0,0,1);
 
    isosurface = new marching_tetrahedron<unstructured_tetrahedra<SPACE>, unstructured_tetrahedra<SPACE> >(*utet, *utet, isovalue);

    ((*isosurface)());
    vertices.assign(isosurface->vertices_begin(), isosurface->vertices_end());
    lookPos = make_float3(0.0f, 0.0f, 0.0f);
    centerPos = make_float3(0.0f, 0.0f, 0.0f);
    for (unsigned int i=0; i<vertices.size(); i++) { centerPos.x += vertices[i].x;  centerPos.y += vertices[i].y;  centerPos.z += vertices[i].z; }
    centerPos.x /= vertices.size();  centerPos.y /= vertices.size();  centerPos.z /= vertices.size(); 
    printf("Center: %f %f %f\n", centerPos.x, centerPos.y, centerPos.z);

    isosurface->useInterop = useInterop;

    #ifdef USE_INTEROP
      if (useInterop)
      {
        for (int i=0; i<4; i++) isosurface->vboResources[i] = vboResources[i];
        isosurface->minIso = minValue;  isosurface->maxIso = maxValue;
      }
    #endif
}




