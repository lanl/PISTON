/*
Copyright (c) 2012, Los Alamos National Security, LLC
All rights reserved.
Copyright 2012. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL),
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

Author: Christopher Sewell, csewell@lanl.gov
This simulation is based on the method by Matt Sottile described here: http://syntacticsalt.com/2011/03/10/functional-flocks/
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

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

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

#include <piston/piston_math.h> 
#include <piston/choose_container.h>
#include <piston/hsv_color_map.h>

#define SPACE thrust::detail::default_device_space_tag
using namespace piston;

#include "flock_sim.h"
#include "glyph.h"
#include "glwindow.h"

//! Number of boids
#define INPUT_SIZE 1024


//==========================================================================
/*! 
    Variable declarations
*/
//==========================================================================

//! Variables for timing the framerate
struct timeval begin, end, diff;
int frameCount;

//! The flock simulation and glyph operators
flock_sim* simulation;
glyph<thrust::device_vector<float3>::iterator, thrust::device_vector<float3>::iterator, thrust::device_vector<float>::iterator,
      thrust::device_vector<float3>::iterator, thrust::device_vector<float3>::iterator, thrust::device_vector<uint3>::iterator >* glyphs;

//! Initial positions and velocities for the boids
thrust::host_vector<float3>   inputPositionsHost;
thrust::device_vector<float3> inputPositions;
thrust::device_vector<float3> inputVelocities;

//! Vertices, normals, colors, vertex indices, and scalars for output
thrust::host_vector<float3>  vertices;
thrust::host_vector<float3>  normals;
thrust::host_vector<float4>  colors; 
thrust::host_vector<uint3>   indices;
thrust::device_vector<float> scalars;

//! Vertices, normals, and vertex indices for the sphere and arrow glyphs
thrust::device_vector<float3>  sphereGlyphVertices;
thrust::device_vector<float3>  sphereGlyphNormals;
thrust::device_vector<uint3>   sphereGlyphIndices;
thrust::device_vector<float3>  arrowGlyphVertices;
thrust::device_vector<float3>  arrowGlyphNormals;
thrust::device_vector<uint3>   arrowGlyphIndices;

//! VTK filters to produce the arrow and sphere glyphs
vtkArrowSource *arrowSource;
vtkSphereSource *sphereSource;
vtkPolyData *spherePoly;
vtkPolyData *arrowPoly;
vtkTriangleFilter *triangleFilter;
vtkPolyDataNormals *normalGenerator;

//! Camera and UI variables
int glyphMode;
bool simPaused;
float3 cameraPos;
float cameraFOV;
int gridSize;

//! Vertex buffer objects used by CUDA interop
#ifdef USE_INTEROP
  GLuint vboBuffers[4];  struct cudaGraphicsResource* vboResources[4];
#endif


//==========================================================================
/*! 
    struct randomInit

    Initialize the vector elements with random values between the min and max
*/
//==========================================================================
struct randomInit : public thrust::unary_function<float3, float3>
{
    float minValue, maxValue;

    __host__ __device__
    randomInit(float minValue, float maxValue) : minValue(minValue), maxValue(maxValue) { };

    __host__ __device__
    float3 operator() (float3 i)
    {
      float3 result;
      result.x = minValue + (maxValue-minValue)*((rand() % 100000)/100000.0);
      result.y = minValue + (maxValue-minValue)*((rand() % 100000)/100000.0);
      result.z = minValue + (maxValue-minValue)*((rand() % 100000)/100000.0);
      return result;
    }
};


//==========================================================================
/*! 
    Extract vertices, normals, and vertex indices from a vtkPolyData instance

    \fn	copyPolyData
*/
//==========================================================================
void copyPolyData(vtkPolyData *polyData, thrust::device_vector<float3> &points, thrust::device_vector<float3> &vectors, thrust::device_vector<uint3> &indexes)
{
    // Extract the vertices and normals and copy to the output vectors
    vtkPoints* pts = polyData->GetPoints();
    vtkFloatArray* verts = vtkFloatArray::SafeDownCast(pts->GetData());
    vtkFloatArray* norms = vtkFloatArray::SafeDownCast(polyData->GetPointData()->GetNormals());
    float3* vData = (float3*)verts->GetPointer(0);
    float3* nData = (float3*)norms->GetPointer(0);
    points.assign(vData, vData+verts->GetNumberOfTuples());
    vectors.assign(nData, nData+norms->GetNumberOfTuples());

    // Extract the vertex indices from the cells and copy to the output vectors
    vtkCellArray* cellArray = polyData->GetPolys();
    vtkIdTypeArray* conn = cellArray->GetData();
    vtkIdType* cData = conn->GetPointer(0);
    for (int i=0; i<3*polyData->GetNumberOfPolys(); i++) cData[i] = cData[(i/3)*4+(i%3)+1];
    thrust::host_vector<uint> indexTemp;
    indexTemp.assign(cData, cData+3*polyData->GetNumberOfPolys());
    uint3* c3Data = (uint3*)(thrust::raw_pointer_cast(&*indexTemp.begin()));
    indexes.assign(c3Data, c3Data+polyData->GetNumberOfPolys());
}


//==========================================================================
/*! 
    Constructor for GLWindow class

    \fn	GLWindow::GLWindow
*/
//==========================================================================
GLWindow::GLWindow(QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    // Start the QT callback timer
    setFocusPolicy(Qt::StrongFocus);
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateGL()));
    timer->start(1);
}


//==========================================================================
/*! 
    Create the flock simulation and glyph operators

    \fn	GLWindow::initializeGL
*/
//==========================================================================
void GLWindow::initializeGL()
{
    // Initialize camera and UI variables
    qrot.set(0,0,0,1);
    frameCount = 0;
    gridSize = 256;
    glyphMode = 0;
    simPaused = false;
    cameraPos = make_float3(0.0f, 0.0f, 1.5*gridSize);
    cameraFOV = 60.0;

    // Set up basic OpenGL state and lighting
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    float white[] = { 0.8, 0.8, 0.8, 1.0 };
    float black[] = { 0.0, 0.0, 0.0, 1.0 };
    float lightPos[] = { 0.0, 0.0, gridSize*1.5, 1.0 };
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

    // Initialize CUDA interop if it is being used
    #ifdef USE_INTEROP
      glewInit();
      cudaGLSetGLDevice(0);
    #endif

    // Initialize boid positions to random values and boid velocities to zero
    inputPositions.resize(INPUT_SIZE); inputPositionsHost.resize(INPUT_SIZE); 
    thrust::transform(inputPositionsHost.begin(), inputPositionsHost.end(), inputPositionsHost.begin(), randomInit(0.0f, 1.0f*gridSize));
    inputPositions = inputPositionsHost;
    thrust::fill(inputVelocities.begin(), inputVelocities.end(), make_float3(0.0f, 0.0f, 0.0f));

    // Set the boundaries of the simulation space
    float3 boundaryMin, boundaryMax;  
    boundaryMin.x = boundaryMin.y = boundaryMin.z = 0.0f;
    boundaryMax.x = boundaryMax.y = boundaryMax.z = gridSize;

    // Create the flock simulation instance
    simulation = new flock_sim(inputPositions, inputVelocities, boundaryMin, boundaryMax, 1.0f, 5.0f, 1.0f, 0.01f, 1.0025f, 30.0f, 5.0f, 30.0f, 4.0f, 10.0f, 0.01f, 0.5f);

    // Use VTK to generate a sphere glyph
    sphereSource = vtkSphereSource::New();
    sphereSource->SetThetaResolution(5);
    sphereSource->SetPhiResolution(5);
    sphereSource->Update();
    spherePoly = vtkPolyData::New();
    spherePoly->ShallowCopy(sphereSource->GetOutput());
    copyPolyData(spherePoly, sphereGlyphVertices, sphereGlyphNormals, sphereGlyphIndices);

    // Use VTK to generate an arrow glyph and its normals
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
    copyPolyData(arrowPoly, arrowGlyphVertices, arrowGlyphNormals, arrowGlyphIndices);

    // Initialize glyph input scalars to the minimum simulation scalar value
    scalars.resize(INPUT_SIZE); 
    thrust::fill(scalars.begin(), scalars.end(), simulation->get_scalar_min());

    // Create the glyph operator instance
    glyphs = new glyph<thrust::device_vector<float3>::iterator, thrust::device_vector<float3>::iterator, thrust::device_vector<float>::iterator,
                       thrust::device_vector<float3>::iterator, thrust::device_vector<float3>::iterator, thrust::device_vector<uint3>::iterator>();
    
    // If using interop, initialize vertex buffer objects
    #ifdef USE_INTEROP
      int numPoints = INPUT_SIZE*std::max(sphereGlyphVertices.size(), arrowGlyphVertices.size());
      glGenBuffers(4, vboBuffers);
      for (int i=0; i<3; i++)
      {
        unsigned int bufferSize = (i == 1) ? numPoints*sizeof(float4) : numPoints*sizeof(float3);
        glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[i]);
        glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_DYNAMIC_DRAW);
      }
      glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[3]);
      glBufferData(GL_ARRAY_BUFFER, numPoints*sizeof(uint3), 0, GL_DYNAMIC_DRAW);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      for (int i=0; i<4; i++)
      {
        cudaGraphicsGLRegisterBuffer(&(vboResources[i]), vboBuffers[i], cudaGraphicsMapFlagsWriteDiscard);
        glyphs->vboResources[i] = vboResources[i];
      }
    #endif

    // Enable OpenGL state for vertex, normal, and color arrays
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
}


//==========================================================================
/*! 
    Update the simulation and graphics

    \fn	GLWindow::paintGL
*/
//==========================================================================
void GLWindow::paintGL()
{
    // Stop the QT callback timer
    timer->stop();

    // Start timing this interval
    if (frameCount == 0) gettimeofday(&begin, 0);

    // Set up the OpenGL state
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Set up the projection and modelview matrices for the view
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(cameraFOV, 1.0f, 1.0f, gridSize*4.0f);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    gluLookAt(cameraPos.x, cameraPos.y, cameraPos.z, cameraPos.x, cameraPos.y, 0.0f, 0.0f, 1.0f, 0.0f);

    // Set up the current rotation and translation
    qrot.getRotMat(rotationMatrix);
    glMultMatrixf(rotationMatrix);
    glTranslatef(-(gridSize-1)/2, -(gridSize-1)/2, -(gridSize-1)/2);

    // If the simulation is not paused, compute the next simulation step, and apply the glyph operator to the result,
    // using either the arrow or sphere glyph
    int curGlyphMode = glyphMode;
    if (!simPaused)
    {
      (*simulation)();
      if (curGlyphMode == 0)
        (*glyphs)(simulation->positions_begin(), simulation->velocities_begin(), simulation->speeds_begin(), 
                  arrowGlyphVertices.begin(), arrowGlyphNormals.begin(), arrowGlyphIndices.begin(),     
                  INPUT_SIZE, arrowGlyphVertices.size(), arrowGlyphIndices.size(), 
                  simulation->get_scalar_min(), simulation->get_scalar_max());
      else 
        (*glyphs)(simulation->positions_begin(), simulation->velocities_begin(), scalars.begin(),
                  sphereGlyphVertices.begin(), sphereGlyphNormals.begin(), sphereGlyphIndices.begin(),     
                  INPUT_SIZE, sphereGlyphVertices.size(), sphereGlyphIndices.size(),
                  simulation->get_scalar_min(), simulation->get_scalar_max());
    }

    // If using interop, render the vertex buffer objects; otherwise, render the arrays
    #ifdef USE_INTEROP
      glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[0]);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboBuffers[3]);
      glVertexPointer(3, GL_FLOAT, 0, 0);
      glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[1]);
      glColorPointer(4, GL_FLOAT, 0, 0);
      glBindBuffer(GL_ARRAY_BUFFER, vboBuffers[2]);
      glNormalPointer(GL_FLOAT, 0, 0);
      int numIndices = INPUT_SIZE;
      if (curGlyphMode == 0) numIndices *= arrowGlyphIndices.size();
      else numIndices *= sphereGlyphIndices.size();
      glDrawElements(GL_TRIANGLES, 3*numIndices, GL_UNSIGNED_INT, 0);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    #else
      normals.assign(glyphs->normals_begin(), glyphs->normals_end());
      indices.assign(glyphs->indices_begin(), glyphs->indices_end());
      vertices.assign(glyphs->vertices_begin(), glyphs->vertices_end());
      colors.assign(thrust::make_transform_iterator(glyphs->scalars_begin(), color_map<float>(simulation->get_scalar_min(), simulation->get_scalar_max())),
    	            thrust::make_transform_iterator(glyphs->scalars_end(), color_map<float>(simulation->get_scalar_min(), simulation->get_scalar_max())));
      glNormalPointer(GL_FLOAT, 0, &normals[0]);
      glColorPointer(4, GL_FLOAT, 0, &colors[0]);
      glVertexPointer(3, GL_FLOAT, 0, &vertices[0]);
      glDrawElements(GL_TRIANGLES, 3*indices.size(), GL_UNSIGNED_INT, &indices[0]);
    #endif

    // Pop this OpenGL view matrix
    glPopMatrix();

    // Periodically output the framerate
    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);
    frameCount++;
    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    if (seconds > 0.5f)
    {
      char title[256];
      sprintf(title, "Flock simulation, fps: %2.2f", float(frameCount)/seconds);
      std::cout << title << std::endl;
      seconds = 0.0f;
      frameCount = 0;
    }

    // Restart the QT callback timer
    timer->start(1);
}


//==========================================================================
/*! 
    Handle window resize event

    \fn	GLWindow::resizeGL
*/
//==========================================================================
void GLWindow::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
}


//==========================================================================
/*! 
    Handle mouse press event

    \fn	GLWindow::mousePressEvent
*/
//==========================================================================
void GLWindow::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}


//==========================================================================
/*! 
    Handle mouse move event to rotate, translate, or zoom

    \fn	GLWindow::mouseMoveEvent
*/
//==========================================================================
void GLWindow::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    // Rotate, zoom, or translate the view
    if (event->buttons() & Qt::LeftButton)
    {
      Quaternion newRotX;
      newRotX.setEulerAngles(-0.2f*dx*3.14159f/180.0f, 0.0f, 0.0f);
      qrot.mul(newRotX);

      Quaternion newRotY;
      newRotY.setEulerAngles(0.0f, 0.0f, -0.2f*dy*3.14159f/180.0f);
      qrot.mul(newRotY);
    }
    else if (event->buttons() & Qt::RightButton)
    {
      cameraFOV += dy/20.0f;
    }
    else if (event->buttons() & Qt::MiddleButton)
    {
      cameraPos.x -= dx/2.0f;
      cameraPos.y += dy/2.0f; 
    }
    lastPos = event->pos();
}


//==========================================================================
/*! 
    Handle keyboard input event

    \fn	GLWindow::keyPressEvent
*/
//==========================================================================
void GLWindow::keyPressEvent(QKeyEvent *event)
{
   // Toggle the glyph type (spheres or arrows)
   if ((event->key() == 'g') || (event->key() == 'G'))
     if (!simPaused) glyphMode = 1 - glyphMode;

   // Pause or resume the simulation
   if ((event->key() == 'p') || (event->key() == 'P'))
     simPaused = !simPaused;
}


