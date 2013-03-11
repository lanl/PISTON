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

#ifdef __APPLE__
    #include <GL/glew.h>
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/glew.h>
    #include <GL/glut.h>
    #include <GL/gl.h>
#endif

#include <QtGui>
#include <QObject>


#ifdef USE_INTEROP
#include <cuda_gl_interop.h>
#endif

#include <piston/piston_math.h> 
#include <piston/choose_container.h>

#define SPACE thrust::detail::default_device_space_tag
using namespace piston;

#include <piston/halo_merge.h>

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

#include "glwindow.h"

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)


struct timeval begin, end, diff;
int frame_count = 0;
int grid_size = 256;
float cameraFOV = 60.0;

halo *haloFinder;

// parameters needed for the halo_finder (look at halo_finder.h for definitions)
float linkLength, max_linkLength;
int   particleSize, rL, np;

bool  haloFound, haloShow;
bool  particleSizeSelected, linkLengthSelected;
float step;

typedef thrust::tuple<float, float, float> Float3;
typedef thrust::device_vector<float>::iterator FloatIterator;
typedef thrust::tuple<FloatIterator, FloatIterator, FloatIterator> Float3IteratorTuple;
typedef thrust::zip_iterator<Float3IteratorTuple> Float3zipIterator;
//-------


GLWindow::GLWindow(QWidget *parent) : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    setFocusPolicy(Qt::StrongFocus);
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateGL()));
    timer->start(1);
}


GLWindow::~GLWindow()
{

}


QSize GLWindow::minimumSizeHint() const
{
    return QSize(100, 100);
}


QSize GLWindow::sizeHint() const
{
    return QSize(1024, 1024);
}


bool GLWindow::initialize(int argc, char *argv[])
{
    return true;
}

void GLWindow::initializeGL()
{
    qrot.set(0,0,0,1);

    // glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    // good old-fashioned fixed function lighting
    float white[] = { 0.8, 0.8, 0.8, 1.0 };
    float black[] = { 0.0, 0.0, 0.0, 1.0 };
    float lightPos[] = { 0.0, 0.0, grid_size*1.5, 1.0 };

    glLightfv(GL_LIGHT0, GL_AMBIENT, white);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, black);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    // Setup the view of the cube.
    glMatrixMode(GL_PROJECTION);
    gluPerspective( cameraFOV, 1.0, 1.0, grid_size*4.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, grid_size*1.5,
              0.0, 0.0, 0.0,
              0.0, 1.0, 0.0);

    particleSizeSelected = false;
    linkLengthSelected   = true;
    haloFound = haloShow = false;
    step = 0.1;

    max_linkLength = 1.5;
    linkLength   = 0.2;
    particleSize = 100;
    np = 256;
    rL = 100;
    int n = 1; // if you want a fraction of the file to load, use this.. 1/n

    char filename[1024];
    sprintf(filename, "%s/sub-24474", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
    std::string format = "csv";
//    sprintf(filename, "%s/256", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
//    std::string format = "cosmo";

    haloFinder = new halo_merge(max_linkLength, filename, format, n, np, rL); // maxLL as a parameter

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
}

void GLWindow::paintGL()
{
    if (frame_count == 0) gettimeofday(&begin, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective( cameraFOV, 1.0, 1.0, grid_size*4.0);

    // set view matrix for 3D scene
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    qrot.getRotMat(rotationMatrix);
    glMultMatrixf(rotationMatrix);

    glTranslatef(-(grid_size-1)/2, -(grid_size-1)/2, -(grid_size-1)/2);

    glBegin(GL_POINTS);

    glColor3f(1, 0, 0);
    int count = 0;

    for(Float3zipIterator it = haloFinder->vertices_begin(); it!=haloFinder->vertices_end(); it++)
    {
      Float3 vertex = *it;

      //set halo color
      if (haloFound && haloShow)
      {
        int haloIndU  = haloFinder->getHaloInd(count++);
		if(haloIndU==-1) continue;

		Float3 color  = haloFinder->getColor(haloIndU);
		glColor3f(thrust::get<0>(color), thrust::get<1>(color), thrust::get<2>(color));
      }
      glVertex3f(thrust::get<0>(vertex), thrust::get<1>(vertex), thrust::get<2>(vertex));
    }

    glEnd();

    glPopMatrix();

    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);
    frame_count++;
    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    if (seconds > 0.5f)
    {
      char title[256];
      sprintf(title, "Halo Finder, fps: %2.2f", float(frame_count)/seconds);
      seconds = 0.0f;
      frame_count = 0;
    }

    timer->start(1);
}


void GLWindow::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
}


void GLWindow::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}


void GLWindow::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    if (event->buttons() & Qt::LeftButton)
    {
      Quaternion newRotX;
      newRotX.setEulerAngles(-0.2*dx*3.14159/180.0, 0.0, 0.0);
      qrot.mul(newRotX);

      Quaternion newRotY;
      newRotY.setEulerAngles(0.0, 0.0, -0.2*dy*3.14159/180.0);
      qrot.mul(newRotY);
    }
    else if (event->buttons() & Qt::RightButton)
    {
      cameraFOV += dy/20.0;
    }
    lastPos = event->pos();
}


void GLWindow::keyPressEvent(QKeyEvent *event)
{
    //toggle between showing halos
    if ((event->key() == 'h') || (event->key() == 'H')) 
    {
      if (!haloShow && !haloFound)
      {
        (*haloFinder)(linkLength, particleSize);
        haloFound = true;
      }
      haloShow = !haloShow;
    }

    //toggle between changing linkLength & particleSize
    if ((event->key() == 't') || (event->key() == 'T')) 
    {
      particleSizeSelected = !particleSizeSelected;
      linkLengthSelected   = !linkLengthSelected;

      std::cout << (linkLengthSelected ? "linkLength Selected": "particleSize Selected") << std::endl;
    }

    if ((event->key() == '+') || (event->key() == '='))
    {
      if (linkLengthSelected) linkLength += step;
      else if (particleSizeSelected) particleSize += step;

      std::cout << "new input..." << std::endl;
      std::cout << "linkLength : " << linkLength << ", particleSize : " << particleSize << std::endl;

      (*haloFinder)(linkLength, particleSize);
    }
    else if ((event->key() == '-') || (event->key() == '_'))
    {
      if (linkLengthSelected) linkLength -= step;
      else if (particleSizeSelected) particleSize -= step;

      std::cout << "new input..." << std::endl;
      std::cout << "linkLength : " << linkLength << ", particleSize : " << particleSize << std::endl;

      (*haloFinder)(linkLength, particleSize);
    }
}


