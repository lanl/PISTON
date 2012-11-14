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
#include <thrust/binary_search.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <float.h>
#include <sys/time.h>

#include "distributedrender.h"
#include <piston/dthrust.h>


DistributedRender::DistributedRender()
{
}


void DistributedRender::setIsovaluePct(float pct)
{
    isovaluePct = pct;
    isovalue = minIso + pct*(maxIso-minIso);
}



void DistributedRender::setZoomLevelPct(float pct)
{
    if (pct > 1.0) pct = 1.0;  if (pct < 0.0) pct = 0.0;
    zoomLevelPct = pct;
    cameraFOV = zoomLevelBase*pct;
}


void DistributedRender::resetView()
{
    qrot.set(0.0f, 0.0f, 0.0f, 1.0f);
    zoomLevelPct = 0.5f;
    cameraFOV = zoomLevelBase*zoomLevelPct;
}


struct timeval begin, end, diff;
float seconds;
void DistributedRender::display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(cameraFOV, 1.0, 1.0f, 4.0f*fabs(cameraZ-center_pos.z));

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

    glEnableClientState(GL_VERTEX_ARRAY);  
    glDisableClientState(GL_COLOR_ARRAY);   
    glEnableClientState(GL_NORMAL_ARRAY);

    glColor3f(1.0f, 0.0f, 0.0f);     
    glNormalPointer(GL_FLOAT, 0, &normals2[0]);
    //glColorPointer(4, GL_FLOAT, 0, &colors[0]);
    glVertexPointer(4, GL_FLOAT, 0, &vertices2[0]);
    glDrawArrays(GL_TRIANGLES, 0, vertices.size());     

    glPopMatrix();
}


void DistributedRender::cleanup()
{
    vertices.clear(); normals.clear(); colors.clear();
}


void DistributedRender::initContour()
{
    
    int commSize;  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
    int commRank;  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));

    if (commRank == 0)
    {
      cayley2 = new tangle_field<SPACE>(GRID_SIZE, GRID_SIZE, GRID_SIZE);
      contour2 = new marching_cube<tangle_field<SPACE>, tangle_field<SPACE> >(*cayley2, *cayley2, 0.46f);

      (*contour2)();
      vertices2.assign(contour2->vertices_begin(), contour2->vertices_end());
      normals2.assign(contour2->normals_begin(), contour2->normals.end());

      thrust::device_vector<int> test1, test2;  test1.resize(24);  test2.resize(12);  for (unsigned int i=0; i<24; i++) test1[i] = i / 3;  
      thrust::upper_bound(test1.begin(), test1.end(), thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(0)+12, test2.begin());
      for (unsigned int i=0; i<test2.size(); i++) std::cout << test2[i] << " ";  std::cout << std::endl;

    }

    cayley = new tangle_field<SPACE>(GRID_SIZE, GRID_SIZE, GRID_SIZE/*, commRank == 0*/);
    contour = new dmarching_cube<tangle_field<SPACE>, tangle_field<SPACE> >(*cayley, *cayley, 0.46f);

    (*contour)();
    
    dthrust::device_to_host(contour->num_total_vertices, contour->vertices, vertices); 
    dthrust::device_to_host(contour->num_total_vertices, contour->normals, normals);  

    int gsize1 = 24;  int gsize2 = 15;
    int lsize1 = gsize1/commSize;  
    thrust::host_vector<int> upinput, upoutput;  thrust::device_vector<int> upinputd, upoutputd;  
    if (commRank == 0) { upinput.resize(gsize1);  for (unsigned int i=0; i<gsize1; i++) upinput[i] = i / 2; }
    dthrust::host_to_device(lsize1, upinput, upinputd);
    dthrust::upper_bound_counting(upinputd.begin(), upinputd.end(), gsize2-1, upoutputd);
    dthrust::device_to_host(upoutputd.size(), upoutputd, upoutput);
    if (commRank == 0) { for (unsigned int i=0; i<upoutput.size(); i++) std::cout << upoutput[i] << " ";  std::cout << std::endl; }


    center_pos = make_float3(0.0f, 0.0f, 0.0f); 
    cameraZ = 5.0f;
    camera_up = make_float3(0.0f, 1.0f, 0.0f);
    zoomLevelBase = 90.0f;
    zoomLevelPct = 0.5f; 
    cameraFOV = zoomLevelPct*zoomLevelBase;
}


void DistributedRender::initGL()
{
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    float white[] = { 0.5, 0.5, 0.5, 1.0 };
    float black[] = { 0.0, 0.0, 0.0, 1.0 };
    float lightPos[] = { GRID_SIZE/2.0f, GRID_SIZE/2.0f, 4.0f*GRID_SIZE, 1.0 };
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
}


void DistributedRender::timeContours()
{
    struct timeval begin, end, diff;
    gettimeofday(&begin, 0);
    int numIters = 10;
    for (int i=0; i<numIters; i++)
    {
      isovalue = minIso; // + ((1.0*i)/(1.0*numIters))*(maxIso - minIso);
      //std::cout << "Generating isovalue " << isovalue << std::endl;
      contour->set_isovalue(isovalue);
      
      (*contour)();
    
      //dthrust::device_to_host(contour->num_total_vertices, contour->vertices, vertices); 
      //dthrust::device_to_host(contour->num_total_vertices, contour->normals, normals);
    }
    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);
    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    std::cout << "contour fps: " << numIters/seconds << std::endl;
}




