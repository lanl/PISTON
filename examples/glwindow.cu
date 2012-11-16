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
//#include <QtOpenGL>
#include <QObject>

#ifdef USE_INTEROP
#include <cuda_gl_interop.h>
#endif

#include <piston/piston_math.h>
#include <piston/choose_container.h>

#define SPACE thrust::detail::default_device_space_tag
using namespace piston;

#include <piston/implicit_function.h>
#include <piston/image3d.h>
#include <piston/marching_cube.h>
#include <piston/util/tangle_field.h>
#include <piston/util/plane_field.h>
#include <piston/util/sphere_field.h>
#include <piston/threshold_geometry.h>
#include <piston/plane_filed_adaptor.h>

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

#include "glwindow.h"


struct timeval begin, end, diff;
int frame_count = 0;
int grid_size = 256;
float cameraFOV = 60.0;
bool wireframe = false;

tangle_field<SPACE>* tangle;
marching_cube<tangle_field<SPACE>, tangle_field<SPACE> > *isosurface;

//plane_field<SPACE>* plane;
//marching_cube<plane_field<SPACE>, tangle_field<SPACE> > *cutplane;
plane_field_adaptor<tangle_field<SPACE> >* plane;
marching_cube<plane_field_adaptor<tangle_field<SPACE> >, tangle_field<SPACE> > *cutplane;

sphere_field<SPACE>* scalar_field;
threshold_geometry<sphere_field<SPACE> >* threshold;

GLuint quads_vbo[3];
struct cudaGraphicsResource *quads_pos_res, *quads_normal_res, *quads_color_res;
unsigned int buffer_size;


thrust::host_vector<float4> vertices;
thrust::host_vector<float3> normals;
thrust::host_vector<float4> colors;

#if USE_INTEROP
void create_vbo()
{
    glGenBuffers(3, quads_vbo);
    int error;

    // initialize vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
    if (glGetError() == GL_OUT_OF_MEMORY) { std::cout << "Out of memory; buffer size too large." << std::endl; exit(-1); }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // register this buffer object with CUDA
    if ((error = cudaGraphicsGLRegisterBuffer(&quads_pos_res, quads_vbo[0],
                                              cudaGraphicsMapFlagsWriteDiscard)) != cudaSuccess) {
	std::cout << "register pos buffer cuda error: " << error << "\n";
    }

    // initialize vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
    if (glGetError() == GL_OUT_OF_MEMORY) { std::cout << "Out of memory; buffer size too large." << std::endl; exit(-1); }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // register this buffer object with CUDA
    if ((error = cudaGraphicsGLRegisterBuffer(&quads_normal_res, quads_vbo[1],
                                              cudaGraphicsMapFlagsWriteDiscard)) != cudaSuccess) {
	std::cout << "register normal buffer cuda error: " << error << "\n";
    }

    // initialize color buffer object
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
    if (glGetError() == GL_OUT_OF_MEMORY) { std::cout << "Out of memory; buffer size too large." << std::endl; exit(-1); }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // register this buffer object with CUDA
    if ((error = cudaGraphicsGLRegisterBuffer(&quads_color_res, quads_vbo[2],
                                     cudaGraphicsMapFlagsWriteDiscard)) != cudaSuccess) {
	std::cout << "register color buffer cuda error: " << error << "\n";
    }
}
#endif

GLWindow::GLWindow(QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
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

struct print_float4
{
    void operator()(float4 pos) {
      std::cout << "("
                << pos.x << ", "
                << pos.y << ", "
                << pos.z << ")" << std::endl;
//        std::cout << (pos.x * pos.x + pos.y * pos.y) << std::endl;
    }
};

void GLWindow::initializeGL()
{
    qrot.set(0,0,0,1);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    // good old-fashioned fixed function lighting
    float white[] = { 0.8, 0.8, 0.8, 1.0 };
    float black[] = { 0.0, 0.0, 0.0, 1.0 };
    float lightPos[] = { 0.0, 0.0, 1.5, 1.0 };

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
    gluPerspective( cameraFOV, 1.0, 1.0, 4.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, 4.0,
              0.0, 0.0, 0.0,
              0.0, 1.0, 0.0);

#ifdef USE_INTEROP
    glewInit();
    cudaGLSetGLDevice(0);
#endif

#ifdef TANGLE
    tangle = new tangle_field<SPACE>(grid_size, grid_size, grid_size);
    isosurface = new marching_cube<tangle_field<SPACE>,  tangle_field<SPACE> >(*tangle, *tangle, 0.3f);
    (*isosurface)();
    buffer_size = thrust::distance(isosurface->vertices_begin(), isosurface->vertices_end())* sizeof(float4);
//    thrust::for_each(isosurface->vertices_begin(),
//                     isosurface->vertices_end(), print_float4());

    std::cout << "buf_size: " << buffer_size << "\n";
#endif

#ifdef CUTPLANE
    tangle = new tangle_field<SPACE>(grid_size, grid_size, grid_size);
//    plane = new plane_field_adaptor<SPACE>(make_float3(0.0f, 0.0f, grid_size/2), make_float3(0.0f, 0.0f, 1.0f), grid_size, grid_size, grid_size);
    plane = new plane_field_adaptor<tangle_field<SPACE> >(*tangle, make_float3(0, 0, 0), make_float3(0, 0, 1));
    cutplane = new marching_cube<plane_field_adaptor<tangle_field<SPACE> >, tangle_field<SPACE> >(*plane, *tangle, 0.2f);
    (*cutplane)();
    buffer_size = thrust::distance(cutplane->vertices_begin(), cutplane->vertices_end())* sizeof(float4);
#endif

#ifdef THRESHOLD
    scalar_field = new sphere_field<SPACE>(grid_size, grid_size, grid_size);
    threshold = new threshold_geometry<sphere_field<SPACE> >(*scalar_field, 4, 1600);
    (*threshold)();
    buffer_size = thrust::distance(threshold->vertices_begin(), threshold->vertices_end())* sizeof(float4);
#endif

#ifdef USE_INTEROP
    create_vbo();
#endif

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
}


void GLWindow::paintGL()
{
    timer->stop();

    if (frame_count == 0) gettimeofday(&begin, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective( cameraFOV, 1.0, 1.0, grid_size*4.0);

    // set view matrix for 3D scene
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    qrot.getRotMat(rotationMatrix);
    glMultMatrixf(rotationMatrix);

//    glTranslatef(-(grid_size-1)/2, -(grid_size-1)/2, -(grid_size-1)/2);
    size_t num_bytes;  GLenum drawType = GL_TRIANGLES;

#ifdef TANGLE
    #ifdef USE_INTEROP
        isosurface->vboResources[0] = quads_pos_res;  isosurface->vboResources[1] = quads_color_res;  isosurface->vboResources[2] = quads_normal_res;
        isosurface->minIso = 31.0f;  isosurface->maxIso = 500.0f;  isosurface->useInterop = true;
        isosurface->vboSize = buffer_size;
    #endif
        (*isosurface)();
    #ifndef USE_INTEROP
        vertices.assign(isosurface->vertices_begin(), isosurface->vertices_end());
        normals.assign(isosurface->normals_begin(), isosurface->normals_end());
        colors.assign(thrust::make_transform_iterator(isosurface->scalars_begin(), color_map<float>(31.0f, 500.0f)),
                      thrust::make_transform_iterator(isosurface->scalars_end(), color_map<float>(31.0f, 500.0f)));
     #endif
#endif

#ifdef CUTPLANE
    #ifdef USE_INTEROP
        cutplane->vboResources[0] = quads_pos_res;  cutplane->vboResources[1] = quads_color_res;  cutplane->vboResources[2] = quads_normal_res;
        cutplane->minIso = 0.0f;  cutplane->maxIso = 1.0f;  cutplane->useInterop = true;
        cutplane->vboSize = buffer_size;
    #endif
    (*cutplane)();
    #ifndef USE_INTEROP
        vertices.assign(cutplane->vertices_begin(), cutplane->vertices_end());
        normals.assign(cutplane->normals_begin(), cutplane->normals_end());
        colors.assign(thrust::make_transform_iterator(cutplane->scalars_begin(), color_map<float>(0.0f, 1.0f)),
                      thrust::make_transform_iterator(cutplane->scalars_end(), color_map<float>(0.0f, 1.0f)));
    #endif
#endif

#ifdef THRESHOLD
    #ifdef USE_INTEROP
        threshold->vboResources[0] = quads_pos_res;  threshold->vboResources[1] = quads_color_res;  threshold->vboResources[2] = quads_normal_res;
        threshold->minThresholdRange = 4.0f;  threshold->maxThresholdRange = 1600.0f;  threshold->useInterop = true;
        threshold->vboSize = buffer_size;
    #endif
    (*threshold)();
    #ifndef USE_INTEROP
        vertices.resize(threshold->vertices_end() - threshold->vertices_begin());
        normals.resize(threshold->normals_end() - threshold->normals_begin());
        thrust::device_vector<float4> device_colors(vertices.size());
//        thrust::copy(thrust::make_transform_iterator(threshold->vertices_begin(), tuple2float4()),
//                     thrust::make_transform_iterator(threshold->vertices_end(), tuple2float4()), vertices.begin());
        thrust::copy(threshold->vertices_begin(),
                     threshold->vertices_end(),
                     vertices.begin());
        thrust::copy(threshold->normals_begin(), threshold->normals_end(), normals.begin());
        thrust::transform(threshold->scalars_begin(), threshold->scalars_end(), device_colors.begin(), color_map<float>(4.0f, 1600.0f));
        colors = device_colors;
    #endif
    drawType = GL_QUADS;
#endif

#ifdef USE_INTEROP
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[0]);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[1]);
    glNormalPointer(GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[2]);
    glColorPointer(4, GL_FLOAT, 0, 0);

    glDrawArrays(drawType, 0, buffer_size/sizeof(float4));
#else
    glColorPointer(4, GL_FLOAT, 0, &colors[0]);
    glNormalPointer(GL_FLOAT, 0, &normals[0]);
    glVertexPointer(4, GL_FLOAT, 0, &vertices[0]);
    glDrawArrays(drawType, 0, vertices.size());
#endif

    glPopMatrix();

    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);
    frame_count++;
    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    if (seconds > 0.5f)
    {
      char title[256];
      sprintf(title, "Marching Cube, fps: %2.2f", float(frame_count)/seconds);
      std::cout << title << std::endl;
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
   if ((event->key() == 'w') || (event->key() == 'W'))
       wireframe = !wireframe;
}


