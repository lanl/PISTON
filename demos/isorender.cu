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

#include "isorender.h"

#define PACKED __attribute__((packed))

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)


struct Rect
{
    int left,top,right,bottom;
};

struct TGAHeader
{
    unsigned char  identsize		;   // size of ID field that follows 18 uint8 header (0 usually)
    unsigned char  colourmaptype	;   // type of colour map 0=none, 1=has palette
    unsigned char  imagetype		;   // type of image 0=none,1=indexed,2=rgb,3=grey,+8=rle packed

    unsigned short colourmapstart	PACKED;   // first colour map entry in palette
    unsigned short colourmaplength	PACKED;   // number of colours in palette
    unsigned char  colourmapbits	;         // number of bits per palette entry 15,16,24,32

    unsigned short xstart		PACKED;   // image x origin
    unsigned short ystart		PACKED;   // image y origin
    unsigned short width		PACKED;   // image width in pixels
    unsigned short height		PACKED;   // image height in pixels
    unsigned char  bits			;         // image bits per pixel 8,16,24,32
    unsigned char  descriptor		;         // image descriptor bits (vh flip bits)

    inline bool IsFlippedHorizontal() const
    {
      return (descriptor & 0x10) != 0;
    }

    inline bool IsFlippedVertical() const
    {
      return (descriptor & 0x20) != 0;
    }
};


IsoRender::IsoRender()
{
    userMode = DEFAULT_MODE;
    animate = false;
    mouse_buttons = 0;
    translate = make_float3(0.0, 0.0, 0.0);
    frameCount = 0;
    lastIsovalue = -9999.9;
    lastPlaneLevel = -9999.9;
    lastThreshold = -9999.9;
    planeLevel = 0.0;
    includePlane = false;
    contours.resize(numDataSets);
    planeContours.resize(numDataSets);
    thresholds.resize(numDataSets);
    constantContours.resize(numDataSets);
    images.resize(numDataSets);
    planeFields.resize(numDataSets);
    polyData.resize(numDataSets);
    readers.resize(numDataSets);
    plyReaders.resize(numDataSets);
    timerTotal = 0; timerCount = 0;
    curFPS = 0.0f;
    for (int i=0; i<numDataSets; i++)
    {
      contours[i] = 0;
      planeContours[i] = 0;
      thresholds[i] = 0;
      constantContours[i] = 0;
      readers[i] = vtkXMLImageDataReader::New();
      polyData[i] = 0;
      plyReaders[i] = vtkPLYReader::New();
    }
}


void IsoRender::setIsovaluePct(float pct)
{
    isovaluePct = pct;
    isovalue = minIso + pct*(maxIso-minIso);

    thresholdPct = pct;
    threshold = minThreshold + pct*(maxThreshold-minThreshold);

    if (userMode == CUT_SURFACE_MODE) setPlaneLevelPct(pct);
}


void IsoRender::setPlaneLevelPct(float pct)
{
    planeLevelPct = pct;
    planeLevel = (-0.95 + (1.0-pct)*1.9)*(planeMax/2.0);
}


void IsoRender::setZoomLevelPct(float pct)
{
    if (pct > 1.0) pct = 1.0;  if (pct < 0.0) pct = 0.0;
    zoomLevelPct = pct;
    cameraFOV = 0.0 + zoomLevelBase*pct;
}


void IsoRender::resetView()
{
    qrot.set(qDefault.x, qDefault.y, qDefault.z, qDefault.w);
    zoomLevelPct = zoomLevelPctDefault;
    cameraFOV = 0.0 + zoomLevelBase*zoomLevelPct;
}


struct timeval begin, end, diff;
float seconds;
void IsoRender::display()
{
    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);
    seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    curFPS = 1.0/seconds;
    //std::cout << "Total fps: " << curFPS << std::endl;
    timerCount++;
    if (timerCount > 50)
    {
      timerTotal += 1.0/seconds;
      //std::cout << "Averages: " << timerCount << ": " << timerTotal/(1.0*timerCount-50.0) << std::endl;
    }
    gettimeofday(&begin, 0);
    //animate = true;
    if ((includeContours) && ((fabs(isovalue - lastIsovalue) > 0.01) || (animate)))
    {
      //std::cout << "Generating isovalue " << isovalue << std::endl;
#ifdef USE_INTEROP
      if (useInterop)
      {
        for (int i=0; i<3; i++) contours[dataSetIndex]->vboResources[i] = vboResources[i];
        for (int i=0; i<3; i++) contours[dataSetIndex]->vboBuffers[i] = vboBuffers[i];
        contours[dataSetIndex]->minIso = minIso;  contours[dataSetIndex]->maxIso = maxIso;
      }
#endif
      float value = isovalue;
      if (animate) value += (rand() % 100)/100.0;
      contours[dataSetIndex]->set_isovalue(value);
      (*(contours[dataSetIndex]))();
      lastIsovalue = isovalue;

      if (!useInterop)
      {
        vertices.assign(contours[dataSetIndex]->vertices_begin(), contours[dataSetIndex]->vertices_end());
        normals.assign(contours[dataSetIndex]->normals_begin(), contours[dataSetIndex]->normals_end());
        colors.assign(thrust::make_transform_iterator(contours[dataSetIndex]->scalars_begin(), color_map<float>(minIso, maxIso)),
                      thrust::make_transform_iterator(contours[dataSetIndex]->scalars_end(), color_map<float>(minIso, maxIso)));
      }
    }

    if ((includePlane) && (fabs(planeLevel - lastPlaneLevel) > 0.01))
    {
#ifdef USE_INTEROP
      if (useInterop)
      {
        for (int i=0; i<3; i++) planeContours[dataSetIndex]->vboResources[i] = planeResources[i];
        for (int i=0; i<3; i++) planeContours[dataSetIndex]->vboBuffers[i] = planeBuffers[i];
        planeContours[dataSetIndex]->minIso = minIso;  planeContours[dataSetIndex]->maxIso = maxIso;  planeContours[dataSetIndex]->colorFlip = useThreshold;
      }
#endif
      planeContours[dataSetIndex]->set_isovalue(planeLevel);
      (*(planeContours[dataSetIndex]))();
      lastPlaneLevel = planeLevel;
      if (!useInterop)
      {
        planeVertices.assign(planeContours[dataSetIndex]->vertices_begin(), planeContours[dataSetIndex]->vertices_end());
        planeNormals.assign(planeContours[dataSetIndex]->normals_begin(), planeContours[dataSetIndex]->normals_end());
        planeColors.assign(thrust::make_transform_iterator(planeContours[dataSetIndex]->scalars_begin(), color_map<float>(minIso, maxIso, useThreshold)),
                           thrust::make_transform_iterator(planeContours[dataSetIndex]->scalars_end(), color_map<float>(minIso, maxIso, useThreshold)));
      }
    }

    if ((includeThreshold) && (fabs(threshold - lastThreshold) > 0.01))
    {
      //std::cout << "Generating threshold " << thresholdFloor << " " << threshold << std::endl;
#ifdef USE_INTEROP
      if (useInterop)
      {
        for (int i=0; i<3; i++) thresholds[dataSetIndex]->vboResources[i] = vboResources[i];
        for (int i=0; i<3; i++) thresholds[dataSetIndex]->vboBuffers[i] = vboBuffers[i];
        thresholds[dataSetIndex]->minThresholdRange = minThreshold;  thresholds[dataSetIndex]->maxThresholdRange = maxThreshold;
      }
#endif
      thresholds[dataSetIndex]->set_threshold_range(thresholdFloor, threshold);
      thresholds[dataSetIndex]->colorFlip = true;
      (*(thresholds[dataSetIndex]))();
      lastThreshold = threshold;
      if (!useInterop)
      {
        thrust::device_vector<float4> device_colors;
        vertices.resize(thresholds[dataSetIndex]->vertices_end() - thresholds[dataSetIndex]->vertices_begin());
        normals.resize(thresholds[dataSetIndex]->normals_end() - thresholds[dataSetIndex]->normals_begin());
        device_colors.resize(thresholds[dataSetIndex]->vertices_end() - thresholds[dataSetIndex]->vertices_begin());
        thrust::copy(thresholds[dataSetIndex]->normals_begin(), thresholds[dataSetIndex]->normals_end(), normals.begin());
        thrust::copy(thresholds[dataSetIndex]->vertices_begin(),
                     thresholds[dataSetIndex]->vertices_end(), vertices.begin());
        thrust::transform(thresholds[dataSetIndex]->scalars_begin(), thresholds[dataSetIndex]->scalars_end(),
                          device_colors.begin(), color_map<float>(minThreshold, maxThreshold, true));
        colors = device_colors;
      }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(cameraFOV, 2.0, zNear, zFar);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    gluLookAt(center_pos.x, center_pos.y, cameraZ,
                  center_pos.x, center_pos.y, center_pos.z,
                  camera_up.x, camera_up.y, camera_up.z);
    glPushMatrix();

    float3 center = make_float3(center_pos.x, center_pos.y, center_pos.z);

    qrot.getRotMat(rotationMatrix);
    glMultMatrixf(rotationMatrix);

    GLfloat matrix[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
    float3 offset = make_float3(matrix[0]*center.x + matrix[1]*center.y + matrix[2]*center.z, matrix[4]*center.x + matrix[5]*center.y + matrix[6]*center.z,
                                matrix[8]*center.x + matrix[9]*center.y + matrix[10]*center.z);
    offset.x = center.x - offset.x; offset.y = center.y - offset.y; offset.z = center.z - offset.z;
    glTranslatef(-offset.x, -offset.y, -offset.z);

    if (includeContours)
    {
      glEnableClientState(GL_VERTEX_ARRAY);

      if (bigDemo)
      {
        glDisableClientState(GL_COLOR_ARRAY);
        color_map<float> isoColor(minIso, maxIso);
        float4 icolor = isoColor(isovalue);
        glColor3f(icolor.x, icolor.y, icolor.z);
      }
      else
      {
        glEnableClientState(GL_COLOR_ARRAY);
      }
      glEnableClientState(GL_NORMAL_ARRAY);

      if (useInterop)
      {
        glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[0]);
        glVertexPointer(4, GL_FLOAT, 0, 0);

        if (!bigDemo)
        {
          glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[1]);
          glColorPointer(4, GL_FLOAT, 0, 0);
        }

        glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[2]);
        glNormalPointer(GL_FLOAT, 0, 0);

        glDrawArrays(GL_TRIANGLES, 0, contours[dataSetIndex]->num_total_vertices);
      }
      else
      {
        glNormalPointer(GL_FLOAT, 0, &normals[0]);
        glColorPointer(4, GL_FLOAT, 0, &colors[0]);
        glVertexPointer(4, GL_FLOAT, 0, &vertices[0]);
        glDrawArrays(GL_TRIANGLES, 0, vertices.size());
      }
    }

    if (includePlane)
    {
      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_COLOR_ARRAY);
      glEnableClientState(GL_NORMAL_ARRAY);

      if (useInterop)
      {
        glBindBuffer(GL_ARRAY_BUFFER, planeBuffers[0]);
        glVertexPointer(4, GL_FLOAT, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, planeBuffers[1]);
        glColorPointer(4, GL_FLOAT, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, planeBuffers[2]);
        glNormalPointer(GL_FLOAT, 0, 0);

        glDrawArrays(GL_TRIANGLES, 0, planeContours[dataSetIndex]->num_total_vertices);
      }
      else
      {
        glNormalPointer(GL_FLOAT, 0, &planeNormals[0]);
        glColorPointer(4, GL_FLOAT, 0, &planeColors[0]);
        glVertexPointer(4, GL_FLOAT, 0, &planeVertices[0]);
        glDrawArrays(GL_TRIANGLES, 0, planeVertices.size());
      }
    }

    if (includeThreshold)
    {
      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_COLOR_ARRAY);
      glEnableClientState(GL_NORMAL_ARRAY);
      if (useInterop)
      {
        glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[0]);
        glVertexPointer(4, GL_FLOAT, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[1]);
        glColorPointer(4, GL_FLOAT, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[2]);
        glNormalPointer(GL_FLOAT, 0, 0);

        glDrawArrays(GL_QUADS, 0, thresholds[dataSetIndex]->num_total_vertices);
      }
      else
      {
        glNormalPointer(GL_FLOAT, 0, &normals[0]);
        glColorPointer(4, GL_FLOAT, 0, &colors[0]);
        glVertexPointer(4, GL_FLOAT, 0, &vertices[0]);
        glDrawArrays(GL_QUADS, 0, vertices.size());
      }
    }

    if (includePolygons)
    {
      polyTriangles = polyData[dataSetIndex]->GetPolys();
      polyTriangles->InitTraversal();

      glDisableClientState(GL_VERTEX_ARRAY);
      glDisableClientState(GL_COLOR_ARRAY);
      glDisableClientState(GL_NORMAL_ARRAY);
      glColor3f(0.5, 0.5, 0.5);

      glPushMatrix();
      glTranslatef(xMax/2.0, yMax/2.0, 0.0);

      glBegin(GL_TRIANGLES);
      for (int i=0; i<polyTriangles->GetNumberOfCells(); i++)
      {
        polyTriangles->GetNextCell(npts, curTriangle);
    	for (int j=0; j<npts; j++)
    	{
    	  double p[3];
          polyData[dataSetIndex]->GetPoint(curTriangle[j], p);
          glVertex3f(polyScale*(p[0]-polyOffset.x), polyScale*(p[1]-polyOffset.y), polyScale*(p[2]-polyOffset.z));
    	}
      }
      glEnd(); gettimeofday(&begin, 0);
      glPopMatrix();
    }

    if (includeConstantContours)
    {
      glEnableClientState(GL_VERTEX_ARRAY);
      glDisableClientState(GL_COLOR_ARRAY);
      glColor3f(0.2, 0.2, 0.2);
      glEnableClientState(GL_NORMAL_ARRAY);

      if (useInterop)
      {
        glBindBuffer(GL_ARRAY_BUFFER, constantBuffers[0]);
        glVertexPointer(4, GL_FLOAT, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, constantBuffers[2]);
        glNormalPointer(GL_FLOAT, 0, 0);

        glDrawArrays(GL_TRIANGLES, 0, numConstantVertices);
      }
      else
      {
        glNormalPointer(GL_FLOAT, 0, &constantNormals[0]);
        glVertexPointer(4, GL_FLOAT, 0, &constantVertices[0]);
        glDrawArrays(GL_TRIANGLES, 0, constantVertices.size());
      }
    }

    glPopMatrix();

    if (showLabels)
    {
      glDisable(GL_LIGHTING);
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();
      glOrtho(0, viewportWidth, 0, viewportHeight, -100000.0, 100000.0);
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
      glDisable(GL_DEPTH_TEST);
      glDepthMask(GL_FALSE);
      glColor3f(1.0, 1.0, 1.0);
      char line[256] = "";

#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA
      sprintf(line, "CUDA Backend");
#endif
#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP
      sprintf(line, "OpenMP Backend");
#endif
      glRasterPos2f(10.0, viewportHeight/8.0);
      for (int c=0; c<strlen(line); c++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, line[c]);

/*#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA
      sprintf(line, "Quadro 6000 GPU (448 cores)");
#endif
#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP
      sprintf(line, "Intel Xeon 2.67 GHz CPU (12 cores)");
#endif
      glRasterPos2f(10.0, viewportHeight/8.0-20);
      for (int c=0; c<strlen(line); c++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, line[c]);*/

      sprintf(line, "Dimensions: %d x %d x %d", xMax-xMin+1, yMax-yMin+1, zMax-zMin+1);
      glRasterPos2f(10.0, viewportHeight/8.0-40);
      for (int c=0; c<strlen(line); c++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, line[c]);

      sprintf(line, "Points: %d", (xMax-xMin+1)*(yMax-yMin+1)*(zMax-zMin+1));
      glRasterPos2f(10.0, viewportHeight/8.0-60);
      for (int c=0; c<strlen(line); c++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, line[c]);

      sprintf(line, "FPS: %.1f", curFPS);
      glRasterPos2f(10.0, viewportHeight/8.0-80);
      for (int c=0; c<strlen(line); c++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, line[c]);

      sprintf(line, "%.1f", maxIso);
      glRasterPos2f(20, 400);
      for (int c=0; c<strlen(line); c++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, line[c]);

      sprintf(line, "%.1f", minIso);
      glRasterPos2f(20, 200);
      for (int c=0; c<strlen(line); c++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, line[c]);

      sprintf(line, "%.1f", (maxIso+minIso)/2.0);
      glRasterPos2f(20, 300);
      for (int c=0; c<strlen(line); c++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, line[c]);

      color_map<float> colorMap(minIso, maxIso, useThreshold);
      float h = 10; float w = 30;
      float x = 100.0;  float y = 200.0;
      glBegin(GL_TRIANGLES);
        for (int i=0; i<20; i++)
        {
    	  float4 mapColor = colorMap(minIso + (i/19.0)*(maxIso-minIso));
    	  glColor4f(mapColor.x, mapColor.y, mapColor.z, mapColor.w);
          glVertex3f(x, y, 0);
          glVertex3f(x+w, y, 0);
          glVertex3f(x, y+h, 0);
          glVertex3f(x+w, y, 0);
          glVertex3f(x+w, y+h, 0);
          glVertex3f(x, y+h, 0);
          y += h;
        }
      glEnd();

      glEnable(GL_LIGHTING);
      glDepthMask(GL_TRUE);
      glEnable(GL_DEPTH_TEST);
      glMatrixMode(GL_PROJECTION);
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
      glPopMatrix();
    }

    if (saveFrames)
    {
      Rect glRect;
      glGetIntegerv( GL_VIEWPORT, (int*)&glRect );
      char screenShotFile[256];
      if (frameCount < 10)
        sprintf(screenShotFile, "%s/Frame00%d.tga", frameDirectory, frameCount);
      else if (frameCount < 100)
        sprintf(screenShotFile, "%s/Frame0%d.tga", frameDirectory, frameCount);
      else 
        sprintf(screenShotFile, "%s/Frame%d.tga", frameDirectory, frameCount);
      screenShot(screenShotFile,glRect.right - glRect.left,glRect.bottom - glRect.top,false);
      std::cout << "Output frame " << frameCount << std::endl;
      frameCount++;
    }
}


void IsoRender::cleanup()
{
#ifdef USE_INTEROP
    if (useInterop)
    {
      printf("Deleting VBO\n");
      if (vboBuffers[0])
      {
        for (int i=0; i<3; i++) cudaGraphicsUnregisterResource(vboResources[i]);
        for (int i=0; i<3; i++)
        {
          glBindBuffer(1, vboBuffers[i]);
      	  glDeleteBuffers(1, &(vboBuffers[i]));
       	  vboBuffers[i] = 0;
        }
      }
      if (planeBuffers[0])
      {
        for (int i=0; i<3; i++) cudaGraphicsUnregisterResource(planeResources[i]);
        for (int i=0; i<3; i++)
        {
          glBindBuffer(1, planeBuffers[i]);
    	  glDeleteBuffers(1, &(planeBuffers[i]));
    	  planeBuffers[i] = 0;
        }
      }
      if (constantBuffers[0])
      {
        for (int i=0; i<3; i++) if (i != 1) cudaGraphicsUnregisterResource(constantResources[i]);
        for (int i=0; i<3; i++)
        {
          if (i == 1) continue;
          glBindBuffer(1, constantBuffers[i]);
    	  glDeleteBuffers(1, &(constantBuffers[i]));
    	  constantBuffers[i] = 0;
        }
      }
    }
    else
#endif
    {
      vertices.clear(); normals.clear(); colors.clear();
      planeVertices.clear(); planeNormals.clear(); planeColors.clear();
      constantVertices.clear(); constantNormals.clear();
    }

    for (int i=0; i<numDataSets; i++)
    {
      if (contours[i]) contours[i]->freeMemory();
      if (thresholds[i]) thresholds[i]->freeMemory();
      if (planeContours[i]) planeContours[i]->freeMemory();
      if (constantContours[i]) constantContours[i]->freeMemory();
    }
}


void IsoRender::initGL(bool aAllowInterop, bool aBigDemo, bool aShowLabels, int aDataSet)
{
    showLabels = aShowLabels;
    if (showLabels)
    {
      int argc = 0; char **argv = 0; glutInit(&argc, argv);
    }

#ifdef USE_INTEROP
    useInterop = aAllowInterop;
#else
    useInterop = false;
#endif
    bigDemo = aBigDemo;

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    float white[] = { 0.5, 0.5, 0.5, 1.0 };
    float black[] = { 0.0, 0.0, 0.0, 1.0 };
    float lightPos[] = { 100.0, 100.0, -100.0, 1.0 };
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
    gluPerspective(cameraFOV, 2.0, zNear, zFar);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(center_pos.x, center_pos.y, cameraZ,
              center_pos.x, center_pos.y, center_pos.z,
              camera_up.x, camera_up.y, camera_up.z);

#ifdef USE_INTEROP
    if (useInterop)
    {
      glewInit();
      cudaGLSetGLDevice(0);

      createBuffers();
    }
#endif

    //printf("Error code: %s\n", cudaGetErrorString(errorCode));
    if (userMode == DEFAULT_MODE) read(aDataSet);
    else read(userFileName, userMode);
}


void IsoRender::timeContours()
{
    contours[dataSetIndex]->useInterop = false;
    struct timeval begin, end, diff;
    gettimeofday(&begin, 0);
    for (int i=0; i<numIters; i++)
    {
      isovalue = minIso; // + ((1.0*i)/(1.0*numIters))*(maxIso - minIso);
      //std::cout << "Generating isovalue " << isovalue << std::endl;
      contours[dataSetIndex]->set_isovalue(isovalue);
      (*(contours[dataSetIndex]))();
    }
    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);
    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    std::cout << "contour fps: " << numIters/seconds << std::endl;
}


void IsoRender::screenShot(std::string fileName, unsigned int width, unsigned int height, bool includeAlpha)
{
    std::cout << "Screen shot" << std::endl;
    unsigned int pixelSize = 3;
    unsigned int pixelSizeBits = 24;
    GLenum pixelFormat = GL_BGR_EXT;

    if (includeAlpha)
    {
      pixelSize = sizeof(unsigned int);
      pixelSizeBits = 32;
      pixelFormat = GL_BGRA_EXT;
    }

    char* pBuffer = new char[pixelSize*width*height ];

    glReadPixels( 0,0,width,height,pixelFormat,GL_UNSIGNED_BYTE,pBuffer );

    TGAHeader tgah;
    memset( &tgah,0,sizeof(TGAHeader) );

    tgah.bits = pixelSizeBits;
    tgah.height = height;
    tgah.width = width;
    tgah.imagetype = 2;

    std::ofstream ofile( fileName.c_str(), std::ios_base::binary );

    ofile.write( (char*)&tgah, sizeof(tgah) );
    ofile.write( pBuffer, pixelSize*width*height );

    ofile.close();

    delete [] pBuffer;
}


void IsoRender::createBuffers()
{
#ifdef USE_INTEROP
    // initialize contour buffer objects
    glGenBuffers(3, vboBuffers);
    for (int i=0; i<3; i++)
    {
      glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[i]);
      glBufferData(GL_ARRAY_BUFFER, BUFFER_SIZE*sizeof(float4), 0, GL_DYNAMIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    for (int i=0; i<3; i++) cudaGraphicsGLRegisterBuffer(&(vboResources[i]), vboBuffers[i], cudaGraphicsMapFlagsWriteDiscard);

    // initialize plane buffer objects
    glGenBuffers(3, planeBuffers);
    for (int i=0; i<3; i++)
    {
      glBindBuffer(GL_ARRAY_BUFFER, planeBuffers[i]);
      glBufferData(GL_ARRAY_BUFFER, BUFFER_SIZE*sizeof(float4), 0, GL_DYNAMIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    for (int i=0; i<3; i++) cudaGraphicsGLRegisterBuffer(&(planeResources[i]), planeBuffers[i], cudaGraphicsMapFlagsWriteDiscard);

    // initialize constant contour buffer objects
    glGenBuffers(3, constantBuffers);
    for (int i=0; i<3; i++)
    {
      if (i == 1) continue;
      glBindBuffer(GL_ARRAY_BUFFER, constantBuffers[i]);
      glBufferData(GL_ARRAY_BUFFER, BUFFER_SIZE*sizeof(float3), 0, GL_DYNAMIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    for (int i=0; i<3; i++) if (i != 1) cudaGraphicsGLRegisterBuffer(&(constantResources[i]), constantBuffers[i], cudaGraphicsMapFlagsWriteDiscard);
#endif
}


void IsoRender::createOperators()
{
    if ((contours[dataSetIndex] == 0) && (thresholds[dataSetIndex] == 0))
    {
      readers[dataSetIndex]->Update();
      output = readers[dataSetIndex]->GetOutput();
      images[dataSetIndex] = new vtk_image3d<SPACE>(output);
    }

    if ((includeContours) && (contours[dataSetIndex] == 0))
    {
      contours[dataSetIndex] = new marching_cube<vtk_image3d<SPACE>, vtk_image3d<SPACE> >(*(images[dataSetIndex]), *(images[dataSetIndex]), isovalue);
      contours[dataSetIndex]->useInterop = useInterop;
      contours[dataSetIndex]->discardMinVals = discardMinVals;
    }

    if ((includePlane) && (planeContours[dataSetIndex] == 0))
    {
      planeFields[dataSetIndex] = new plane_field_adaptor<vtk_image3d<SPACE> >(*images[dataSetIndex],
                                                                 make_float3((xMax+xMin+1)/2.0, (yMax+yMin+1)/2.0, (zMax+zMin+1)/2.0),
//                                                                               make_float3(0, 0, 0),
                                                                 plane_normal);//, xMax-xMin+1, yMax-yMin+1, zMax-zMin+1);
      planeContours[dataSetIndex] = new marching_cube<plane_field_adaptor<vtk_image3d<SPACE> >,
	      vtk_image3d<SPACE> >(*(planeFields[dataSetIndex]), *(images[dataSetIndex]), isovalue);
      planeContours[dataSetIndex]->useInterop = useInterop;
    }

    if (includeThreshold && (thresholds[dataSetIndex] == 0))
    {
      thresholds[dataSetIndex] = new threshold_geometry<vtk_image3d<SPACE> >(*(images[dataSetIndex]), thresholdFloor, threshold);
      thresholds[dataSetIndex]->useInterop = useInterop;
    }

    if ((includeConstantContours) && (constantContours[dataSetIndex] == 0))
    {
      if (useContours) constantContours[dataSetIndex] = contours[dataSetIndex];
      else constantContours[dataSetIndex] = new marching_cube<vtk_image3d<SPACE>, vtk_image3d<SPACE> >(*(images[dataSetIndex]), *(images[dataSetIndex]), isovalue);
    }

    if (includeConstantContours)
    {
#ifdef USE_INTEROP
      if (useInterop)
      {
        for (int i=0; i<3; i++) if (i != 1) constantContours[dataSetIndex]->vboResources[i] = constantResources[i];
        for (int i=0; i<3; i++) if (i != 1) constantContours[dataSetIndex]->vboBuffers[i] = constantBuffers[i];
	constantContours[dataSetIndex]->vboResources[1] = 0;
	constantContours[dataSetIndex]->vboSize = 0;
	constantContours[dataSetIndex]->useInterop = useInterop;
      }
#endif
      constantContours[dataSetIndex]->discardMinVals = false;
      constantContours[dataSetIndex]->set_isovalue(-99999.9);
      constantContours[dataSetIndex]->discardMinVals = false;
      (*(constantContours[dataSetIndex]))();
      if (!useInterop)
      {
        constantVertices.assign(constantContours[dataSetIndex]->vertices_begin(), constantContours[dataSetIndex]->vertices_end());
	    constantNormals.assign(constantContours[dataSetIndex]->normals_begin(), constantContours[dataSetIndex]->normals_end());
      }
      numConstantVertices = constantContours[dataSetIndex]->num_total_vertices;
      if (!useContours) constantContours[dataSetIndex]->freeMemory(false);
      constantContours[dataSetIndex]->discardMinVals = true;
    }

#ifdef USE_INTEROP
    if (includeContours) contours[dataSetIndex]->vboSize = BUFFER_SIZE;
    if (includePlane) planeContours[dataSetIndex]->vboSize = BUFFER_SIZE;
    if (includeThreshold) thresholds[dataSetIndex]->vboSize = BUFFER_SIZE;
#endif

}


int IsoRender::read(char* aFileName, int aMode)
{
    dataSetIndex = 1;  numIters = 0;  saveFrames = 0;

    char filename[1024];
    sprintf(filename, "%s/%s", STRINGIZE_VALUE_OF(DATA_DIRECTORY), userFileName);

    int fileFound = readers[dataSetIndex]->CanReadFile(filename);
    if (fileFound == 0) sprintf(filename, userFileName);
    readers[dataSetIndex]->SetFileName(filename);

    readers[dataSetIndex]->Update();
    output = readers[dataSetIndex]->GetOutput();

    int dims[3];  output->GetDimensions(dims);
    xMin = yMin = zMin = 0;  xMax = dims[0]-1;  yMax = dims[1]-1;  zMax = dims[2]-1;
    double bounds[6];  output->GetBounds(bounds);

    NPoints = (xMax - xMin + 1) * (yMax - yMin + 1) * (zMax - zMin + 1);
    center_pos = make_float3((bounds[0]+bounds[1])/2.0f, (bounds[2]+bounds[3])/2.0f, (bounds[4]+bounds[5])/2.0f);

    if (userRange)
    {
      minIso = userMin;  maxIso = userMax;  minThreshold = thresholdFloor = userMin;  maxThreshold = userMax;
    }
    else
    {
      float* rawData = (float*)(output->GetScalarPointer());
      float minVal = FLT_MAX;  float maxVal = FLT_MIN;
      for (int i=0; i<NPoints; i++)
      {
        if (rawData[i] < minVal) minVal = rawData[i];
	if (rawData[i] > maxVal) maxVal = rawData[i];
      }
      minIso = minVal + (0.01*(maxVal - minVal));  maxIso = maxVal - (0.01*(maxVal - minVal));
      minThreshold = thresholdFloor = minIso;  maxThreshold = maxIso;
    }
    isovaluePct = 0.5;

    discardMinVals = true;
    plane_normal.x = 0.0;  plane_normal.y = 0.0;  plane_normal.z = 1.0;  planeLevelPct = 0.5;
    zoomLevelPctDefault = 0.5;  cameraFOV = 36.0;  cameraZ = 3.0*std::max(std::max(bounds[1]-bounds[0], bounds[3]-bounds[2]), bounds[5]-bounds[4]);
    zFar = 2.0*cameraZ;  zNear = zFar/10.0;
    planeMax = (xMax-xMin)*plane_normal.x + (yMax-yMin)*plane_normal.y + (zMax-zMin)*plane_normal.z;

    zoomLevelBase = cameraFOV;
    qDefault.set(0, 0, 0, 1);  qrot.set(qDefault.x, qDefault.y, qDefault.z, qDefault.w);
    zoomLevelPct = zoomLevelPctDefault;
    isovalue = minIso;  threshold = minThreshold;

    cameraFOV = zoomLevelBase*zoomLevelPct;  camera_up = make_float3(0,1,0);

    includeContours = (userMode == ISOSURFACE_MODE);  includeThreshold = (userMode == THRESHOLD_MODE);  includePlane = (userMode == CUT_SURFACE_MODE);
    includeConstantContours = false;  includePolygons = false;
    useContours = includeContours; useThreshold = includeThreshold; useConstantContours = includeConstantContours;

    createOperators();

    lastIsovalue = -9999.9;
    lastPlaneLevel = -9999.9;
    lastThreshold = -9999.9;
    std::cout << "Read user file " << filename << std::endl;

    return 0;
}


int IsoRender::read(int aDataSetIndex, int aNumIters, int aSaveFrames, char* aFrameDirectory)
{
    dataSetIndex = aDataSetIndex;
    numIters = aNumIters;
    saveFrames = aSaveFrames;
    if (saveFrames) strcpy(frameDirectory, aFrameDirectory);

    char metafile[1024]; char fname[1024]; char pname[1024]; char dtag[1024]; float qx, qy, qz, qw;
    fname[0] = 0; pname[0] = 0;
    sprintf(metafile, "%s/dataset%d.txt", STRINGIZE_VALUE_OF(DATA_DIRECTORY), dataSetIndex);

    std::string line, tag;
    std::ifstream myfile(metafile);
    if (myfile.is_open())
    {
      while (myfile.good())
      {
        getline (myfile,line);
        std::stringstream lineStream(line);
        lineStream >> tag;

        if (tag.compare("data") == 0)       sscanf(line.c_str(), "%s %s", dtag, fname);
        if (tag.compare("dimensions") == 0) sscanf(line.c_str(), "%s %d %d %d %d %d %d", dtag, &xMin, &xMax, &yMin, &yMax, &zMin, &zMax);
        if (tag.compare("polys") == 0)      sscanf(line.c_str(), "%s %s", dtag, pname);
        if (tag.compare("isovalues") == 0)  sscanf(line.c_str(), "%s %f %f %f", dtag, &minIso, &maxIso, &isovaluePct);
        if (tag.compare("thresholds") == 0) sscanf(line.c_str(), "%s %f %f %f", dtag, &minThreshold, &maxThreshold, &thresholdFloor);
        if (tag.compare("zoom") == 0)       sscanf(line.c_str(), "%s %f %f %f", dtag, &cameraFOV, &cameraZ, &zoomLevelPctDefault);
        if (tag.compare("quaternion") == 0) sscanf(line.c_str(), "%s %f %f %f %f", dtag, &qx, &qy, &qz, &qw);
        if (tag.compare("plane") == 0)      sscanf(line.c_str(), "%s %f %f %f %f", dtag, &(plane_normal.x), &(plane_normal.y), &(plane_normal.z), &planeLevelPct);
        if (tag.compare("include") == 0)    sscanf(line.c_str(), "%s %d %d %d %d", dtag, &includeContours, &includePlane, &includeThreshold, &includeConstantContours);
        if (tag.compare("discard") == 0)    sscanf(line.c_str(), "%s %d", dtag, &discardMinVals);
      }
      myfile.close();
    }
    else cout << "Unable to open file";

    char filename[1024];
    sprintf(filename, "%s/%s", STRINGIZE_VALUE_OF(DATA_DIRECTORY), fname);
    readers[dataSetIndex]->SetFileName(filename);

    readers[dataSetIndex]->Update();
    output = readers[dataSetIndex]->GetOutput();
    double bounds[6];  output->GetBounds(bounds);

    NPoints = (xMax - xMin + 1) * (yMax - yMin + 1) * (zMax - zMin + 1);
    center_pos = make_float3((bounds[0]+bounds[1])/2.0f, (bounds[2]+bounds[3])/2.0f, (bounds[4]+bounds[5])/2.0f);
    planeMax = (xMax-xMin)*plane_normal.x + (yMax-yMin)*plane_normal.y + (zMax-zMin)*plane_normal.z;

    zoomLevelBase = cameraFOV;  zNear = 200.0;  zFar = 4000.0;
    qDefault.set(qx, qy, qz, qw);  qrot.set(qDefault.x, qDefault.y, qDefault.z, qDefault.w);
    zoomLevelPct = zoomLevelPctDefault;
    isovalue = minIso;  threshold = minThreshold;

    cameraFOV = zoomLevelBase*zoomLevelPct;  camera_up = make_float3(0,1,0);
    useContours = includeContours; useThreshold = includeThreshold; useConstantContours = includeConstantContours;
    includePolygons = (pname[0] != 0);

    if (includePolygons)
    {
      sprintf(filename, "%s/%s", STRINGIZE_VALUE_OF(DATA_DIRECTORY), pname);
      plyReaders[dataSetIndex]->SetFileName(filename);
      if ((contours[dataSetIndex] == 0) && (thresholds[dataSetIndex] == 0))
      {
        plyReaders[dataSetIndex]->Update();
        polyData[dataSetIndex] = plyReaders[dataSetIndex]->GetOutput();

        polyAvgX = polyAvgY = polyAvgZ = 0.0;
        polyTriangles = polyData[dataSetIndex]->GetPolys();
        ncells = polyTriangles->GetNumberOfCells();
        for (int i=0; i<ncells; i++)
        {
          polyTriangles->GetNextCell(npts, curTriangle);
          for (int j=0; j<npts; j++)
          {
            double p[3];
            polyData[dataSetIndex]->GetPoint(curTriangle[j], p);
            polyAvgX += p[0]; polyAvgY += p[1]; polyAvgZ += p[2];
          }
        }
        polyOffset.x = polyAvgX/(3.0*ncells);
        polyOffset.y = polyAvgY/(3.0*ncells);
        polyOffset.z = 0.0;
        polyScale = 0.25;
      }
    }

    createOperators();

    lastIsovalue = -9999.9;
    lastPlaneLevel = -9999.9;
    lastThreshold = -9999.9;
    std::cout << "Read file " << dataSetIndex << std::endl;

    return 0;
}
