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
#include <QtOpenGL>
#include <QObject>

#include <cuda_gl_interop.h>

#include <vtkXMLImageDataReader.h>

#include <cutil_math.h>
#include <piston/choose_container.h>

#define SPACE thrust::detail::default_device_space_tag
using namespace piston;

#include <piston/util/plane_field.h>
#include <piston/vtk_image3d.h>
#include <piston/marching_cube.h>

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
bool wireframe = false;

vtk_image3d<int, float, SPACE>* image;
plane_field<int, float, SPACE>* plane;
marching_cube<plane_field<int, float, SPACE>, vtk_image3d<int, float, SPACE> > *isosurface;

GLuint quads_vbo[3];
struct cudaGraphicsResource *quads_pos_res, *quads_normal_res, *quads_color_res;
unsigned int buffer_size;


void create_vbo()
{
    glGenBuffers(3, quads_vbo);
    int error;

    //std::cout << "number of vertices: " << thrust::distance(isosurface_p->vertices_begin(), isosurface_p->vertices_end()) << std::endl;
    buffer_size = thrust::distance(isosurface->vertices_begin(), isosurface->vertices_end())* sizeof(float4);

    // initialize vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // register this buffer object with CUDA
    if ((error = cudaGraphicsGLRegisterBuffer(&quads_pos_res, quads_vbo[0],
                                              cudaGraphicsMapFlagsWriteDiscard)) != cudaSuccess) {
	std::cout << "register pos buffer cuda error: " << error << "\n";
    }

    // initialize vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // register this buffer object with CUDA
    if ((error = cudaGraphicsGLRegisterBuffer(&quads_normal_res, quads_vbo[1],
                                              cudaGraphicsMapFlagsWriteDiscard)) != cudaSuccess) {
	std::cout << "register normal buffer cuda error: " << error << "\n";
    }

    // initialize color buffer object
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // register this buffer object with CUDA
    if ((error = cudaGraphicsGLRegisterBuffer(&quads_color_res, quads_vbo[2],
                                     cudaGraphicsMapFlagsWriteDiscard)) != cudaSuccess) {
	std::cout << "register color buffer cuda error: " << error << "\n";
    }
}


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
    return QSize(100, 50);
}


QSize GLWindow::sizeHint() const
{
    return QSize(2048, 1024);
}


bool GLWindow::initialize(int argc, char *argv[])
{
    if (argc < 2) return false;
    sprintf(fileName, argv[1]);
    return true;
}


void GLWindow::initializeGL()
{
    glewInit();
    cudaGLSetGLDevice(0);

    vtkXMLImageDataReader *reader = vtkXMLImageDataReader::New();
    char fname[1024];
    sprintf(fname, "%s/%s", STRINGIZE_VALUE_OF(DATA_DIRECTORY), fileName);
    int fileFound = reader->CanReadFile(fname);
    if (fileFound == 0) sprintf(fname, fileName);
    reader->SetFileName(fname);
    reader->Update();

    vtkImageData *vtk_image = reader->GetOutput();

    std::cout << "Size: " << vtk_image->GetDimensions()[0] << " " << vtk_image->GetDimensions()[1] << " " << vtk_image->GetDimensions()[2] << std::endl;
    image = new vtk_image3d<int, float, SPACE>(vtk_image);
    plane = new plane_field<int, float, SPACE>(make_float3(0.0f, 0.0f, 0.0f),
                                               make_float3(0.0f, 0.0f, 1.0f),
                                               vtk_image->GetDimensions()[0],
                                               vtk_image->GetDimensions()[1],
                                               vtk_image->GetDimensions()[2]);
    isosurface = new marching_cube<plane_field<int, float, SPACE>, vtk_image3d<int, float, SPACE> >(*plane, *image, 0.0f);

    (*isosurface)();

    create_vbo();

    qrot.set(0,0,0,1);
    grid_size = vtk_image->GetDimensions()[1];

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    // good old-fashioned fixed function lighting
    float white[] = { 0.8, 0.8, 0.8, 1.0 };
    float lightPos[] = { 100.0, 100.0, -100.0, 1.0 };

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100);

    glLightfv(GL_LIGHT0, GL_AMBIENT, white);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);

    // Setup the view of the cube.
    glMatrixMode(GL_PROJECTION);
    gluPerspective( cameraFOV, 1.0, 1.0, grid_size*4.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt((grid_size-1)/2, (grid_size-1)/2, grid_size*1.5,
              (grid_size-1)/2, (grid_size-1)/2, 0.0,
              0.0, 1.0, 0.0);

    // enable vertex and normal arrays
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
}


void GLWindow::paintGL()
{
    timer->stop();

    if (frame_count == 0) gettimeofday(&begin, 0);

    (*isosurface)();

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

    float3 center = make_float3((grid_size-1)/2, (grid_size-1)/2, 0.0);
    GLfloat matrix[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
    float3 offset = make_float3(matrix[0]*center.x + matrix[1]*center.y + matrix[2]*center.z, matrix[4]*center.x + matrix[5]*center.y + matrix[6]*center.z,
                                matrix[8]*center.x + matrix[9]*center.y + matrix[10]*center.z);
    offset.x = center.x - offset.x; offset.y = center.y - offset.y; offset.z = center.z - offset.z;
    glTranslatef(-offset.x, -offset.y, -offset.z);

    float4 *raw_ptr;
    size_t num_bytes;

    cudaGraphicsMapResources(1, &quads_pos_res, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&raw_ptr, &num_bytes, quads_pos_res);

    thrust::copy(isosurface->vertices_begin(),
                 isosurface->vertices_end(),
                 thrust::device_ptr<float4>(raw_ptr));

    cudaGraphicsUnmapResources(1, &quads_pos_res, 0);
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[0]);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    float3 *normal;
    cudaGraphicsMapResources(1, &quads_normal_res, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&normal, &num_bytes, quads_normal_res);
    thrust::copy(isosurface->normals_begin(),
                 isosurface->normals_end(),
                 thrust::device_ptr<float3>(normal));
    cudaGraphicsUnmapResources(1, &quads_normal_res, 0);
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[1]);
    glNormalPointer(GL_FLOAT, 0, 0);

    cudaGraphicsMapResources(1, &quads_color_res, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&raw_ptr, &num_bytes, quads_color_res);
    thrust::transform(isosurface->scalars_begin(), isosurface->scalars_end(),
                      thrust::device_ptr<float4>(raw_ptr),
                      color_map<float>(31.0f, 500.0f));
    cudaGraphicsUnmapResources(1, &quads_color_res, 0);
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[2]);
    glColorPointer(4, GL_FLOAT, 0, 0);

    glDrawArrays(GL_TRIANGLES, 0, buffer_size/sizeof(float4));

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


