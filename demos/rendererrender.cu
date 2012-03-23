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

#include "rendererrender.h"

//#define TANGLE_EXAMPLE
#define RTI_EXAMPLE
#define ORTHO

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


RendererRender::RendererRender()
{
    mouse_buttons = 0;
    rcnt = 0;
    translate = make_float3(0.0, 0.0, 0.0);
    grid_size = 256;
    viewportWidth = 2*grid_size;  viewportHeight = 2*grid_size;
    //qDefault.set(1.0f, 1.0f, 0.0f, 1.0f); qDefault.normalize();
    qDefault.set(-0.27, -0.02, -0.71, 0.63); //0.0f, 0.0f, 0.0f, 1.0f);
    qDefault.normalize();
    resetView();
}


void RendererRender::setZoomLevelPct(float pct)
{
    if (pct > 1.0) pct = 1.0;  if (pct < 0.0) pct = 0.0;
    zoomLevelPct = pct;
    cameraFOV = 0.0 + zoomLevelBase*pct;
}


void RendererRender::resetView()
{
    qrot.set(qDefault.x, qDefault.y, qDefault.z, qDefault.w);
    zoomLevelPct = zoomLevelPctDefault;
    cameraFOV = 0.0 + zoomLevelBase*zoomLevelPct;
}


void RendererRender::display()
{
    struct timeval begin, end, diff;
    float seconds;
    gettimeofday(&begin, 0);

    //Quaternion newRotX;
    //newRotX.setEulerAngles(-0.2*50*3.14159/180.0, 0.0, 0.0);
    //qrot.mul(newRotX);
    qrot.getRotMat(rotationMatrix);
    float3 center;  center.x = center.y = center.z = grid_size/2;
    float3 offset = make_float3(rotationMatrix[0]*center.x + rotationMatrix[1]*center.y + rotationMatrix[2]*center.z,
                                rotationMatrix[4]*center.x + rotationMatrix[5]*center.y + rotationMatrix[6]*center.z,
                                rotationMatrix[8]*center.x + rotationMatrix[9]*center.y + rotationMatrix[10]*center.z);
    offset.x = center.x - offset.x; offset.y = center.y - offset.y; offset.z = center.z - offset.z;

    isovalue += isoInc;
    if (isovalue > isoMax) { isovalue = isoMax; isoInc = -isoInc; }
    if (isovalue < isoMin) { isovalue = isoMin; isoInc = -isoInc; }
    printf("Isovalue: %f\n", isovalue);

#ifdef RTI_EXAMPLE
    (*isosurface2)();
    isosurface2->set_isovalue(isovalue);
    inputVertices.assign(isosurface2->vertices_begin(), isosurface2->vertices_end());
    inputNormals.assign(isosurface2->normals_begin(), isosurface2->normals_end());
    inputColors.assign(thrust::make_transform_iterator(isosurface2->scalars_begin(), color_map<float>(31.0f, 500.0f)),
                       thrust::make_transform_iterator(isosurface2->scalars_end(), color_map<float>(31.0f, 500.0f)));
#endif

#ifdef TANGLE_EXAMPLE
    (*isosurface)();
    inputVertices.assign(isosurface->vertices_begin(), isosurface->vertices_end());
    inputNormals.assign(isosurface->normals_begin(), isosurface->normals_end());
    inputColors.assign(thrust::make_transform_iterator(isosurface->scalars_begin(), color_map<float>(31.0f, 500.0f)),
                       thrust::make_transform_iterator(isosurface->scalars_end(), color_map<float>(31.0f, 500.0f)));
#endif

    renders->update(inputVertices.begin(), inputNormals.begin(), inputColors.begin(), inputVertices.size());

    inputVerticesHost = inputVertices;
    inputNormalsHost = inputNormals;
    inputColorsHost = inputColors;

#ifdef ORTHO
    renders->setOrtho(0.0, grid_size, 0.0, grid_size, -2000.0f, 2000.0f);
    renders->setRot(rotationMatrix);
    renders->translate(-offset.x, -offset.y, -offset.z);
    //renders->setOrtho(-10.0f, 10.0f, -10.0f, 10.0f, -2000.0f, 2000.0f);
#else
    renders->setPerspective(cameraFOV, viewportWidth/viewportHeight, 1.0f, 5.0f*grid_size);
    renders->setLookAt(make_float3(0,0,4.0f*grid_size), make_float3(0,0,0), make_float3(0,1,0));

    renders->rotate(rotationMatrix);
    renders->translate(-grid_size/2, -grid_size/2, -grid_size/2);
#endif
    renders->setLightProperties(make_float3(0.5f, 0.5f, 0.5f), make_float3(0.5f, 0.5f, 0.5f), 1.0f, 0.0f, 0.0f, make_float4(0.0f, 0.0f, 10000.0f, 1.0f));

    (*(renders))();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

#ifdef ORTHO
    glOrtho(-0.0, grid_size, 0.0, grid_size, -2000.0f, 2000.0f);
    //glOrtho(-10.0f, 10.0f, -10.0f, 10.0f, -2000.0f, 2000.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMultMatrixf(rotationMatrix);
    glTranslatef(-offset.x, -offset.y, -offset.z);

    //glDisable(GL_LIGHTING);
#else
    gluPerspective(cameraFOV, viewportWidth/viewportHeight, 1.0f, 5.0f*grid_size);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0,0,4.0f*grid_size,0,0,0,0,1,0);

    glMultMatrixf(rotationMatrix);

    glTranslatef(-grid_size/2, -grid_size/2, -grid_size/2);
#endif

    glPushMatrix();

    glDisable(GL_CULL_FACE);

    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    glNormalPointer(GL_FLOAT, 0, &inputNormalsHost[0]);
    glColorPointer(4, GL_FLOAT, 0, &inputColorsHost[0]);
    glVertexPointer(4, GL_FLOAT, 0, &inputVerticesHost[0]);
    glDrawArrays(GL_TRIANGLES, 0, inputVerticesHost.size());

    //if (rcnt == 0)
    //{
      //GLdouble mat1[16];
      //glGetDoublev(GL_PROJECTION_MATRIX,mat1);
      //GLdouble mat2[16];
      //glGetDoublev(GL_MODELVIEW_MATRIX,mat2);

      //std::cout << "OpenGL matrices" << std::endl;
      //for (unsigned int i=0; i<16; i++)  std::cout << mat1[(i%4)*4+(i/4)] << " ";  std::cout << std::endl;
      //for (unsigned int i=0; i<16; i++)  std::cout << mat2[(i%4)*4+(i/4)] << " ";  std::cout << std::endl;
    //}

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glPopMatrix();

    char fname[128];  sprintf(fname, "test%d.tga", rcnt); //if (rcnt == 0)
    screenShot(fname, viewportWidth, viewportHeight, true);
    rcnt++;

    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);
    seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    std::cout << "Total seconds: " << rcnt << " " << seconds << std::endl;

    //if (rcnt == 4) exit(-1);
}


void RendererRender::screenShot(std::string fileName, unsigned int width, unsigned int height, bool includeAlpha)
{
    std::cout << "Saving file" << std::endl;
    unsigned int pixelSize = 3;
    unsigned int pixelSizeBits = 24;
    GLenum pixelFormat = GL_BGR_EXT;

    if (includeAlpha)
    {
      pixelSize = sizeof(unsigned int);
      pixelSizeBits = 32;
      pixelFormat = GL_BGRA_EXT;
    }

    TGAHeader tgah;
    memset( &tgah,0,sizeof(TGAHeader) );

    tgah.bits = pixelSizeBits;
    tgah.height = height;
    tgah.width = width;
    tgah.imagetype = 2;

    std::ofstream ofile( fileName.c_str(), std::ios_base::binary );

    ofile.write( (char*)&tgah, sizeof(tgah) );
    thrust::host_vector<char> hostFrame;
    hostFrame.assign(renders->frame_begin(), renders->frame_end());
    ofile.write( &hostFrame[0], pixelSize*width*height );

    ofile.close();
}


void RendererRender::cleanup()
{

}


void RendererRender::initGL(bool aAllowInterop)
{
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    float white[] = { 0.5, 0.5, 0.5, 1.0 };
    float black[] = { 0.0, 0.0, 0.0, 1.0 };
    float lightPos[] = { 0.0, 0.0, 10000.0, 1.0 };
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

    //printf("Error code: %s\n", cudaGetErrorString(errorCode));
    read();
}


int RendererRender::read()
{
#ifdef RTI_EXAMPLE
    reader = vtkXMLImageDataReader::New();
    char filename[1024];
    sprintf(filename, "%s/rti256.vti", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
    int fileFound = reader->CanReadFile(filename);
    if (fileFound == 0) sprintf(filename, "rti256.vti");
    reader->SetFileName(filename);

    reader->Update();
    output = reader->GetOutput();
    image = new vtk_image3d<int, float, SPACE>(output);
    isovalue = 40.0f;  isoMax = 480.0f;  isoMin = 35.0f;  isoInc = 20.0f;
    isosurface2 = new marching_cube<vtk_image3d<int, float, SPACE>, vtk_image3d<int, float, SPACE> >(*image, *image, isovalue);
    isosurface2->useInterop = false;
    isosurface2->discardMinVals = true;
    (*isosurface2)();
#endif

#ifdef TANGLE_EXAMPLE
    tangle = new tangle_field<int, float, SPACE>(grid_size, grid_size, grid_size);
    isovalue = 0.2f;  isoMax = 0.9f;  isoMin = 0.1f;  isoInc = 0.1f;
    isosurface = new marching_cube<tangle_field<int, float, SPACE>,  tangle_field<int, float, SPACE> >(*tangle, *tangle, isovalue);
    (*isosurface)();
#endif

    /*inputVertices.push_back(make_float4(10.0f,  10.0f, 10.0f, 1.0f));  inputColors.push_back(make_float4(0.0f, 0.0f, 1.0f, 1.0f));
    inputVertices.push_back(make_float4(150.0f, 11.0f, 10.0f, 1.0f));  inputColors.push_back(make_float4(0.0f, 0.0f, 1.0f, 1.0f));
    inputVertices.push_back(make_float4(11.0f, 150.0f, 10.0f, 1.0f));  inputColors.push_back(make_float4(0.0f, 0.0f, 1.0f, 1.0f));

    inputVertices.push_back(make_float4(10.0f,  10.0f,  0.0f, 1.0f));  inputColors.push_back(make_float4(1.0f, 0.0f, 0.0f, 1.0f));
    inputVertices.push_back(make_float4(150.0f, 11.0f,  0.0f, 1.0f));  inputColors.push_back(make_float4(1.0f, 0.0f, 0.0f, 1.0f));
    inputVertices.push_back(make_float4(11.0f, 150.0f,  0.0f, 1.0f));  inputColors.push_back(make_float4(1.0f, 0.0f, 0.0f, 1.0f));*/

    inputVertices.push_back(make_float4(-3.0f, 3.0f, 0.0f, 1.0f));
    inputVertices.push_back(make_float4(-4.0f, 4.0f, 0.0f, 1.0f));
    inputVertices.push_back(make_float4(-5.0f, 2.0f, 0.0f, 1.0f));

    inputVertices.push_back(make_float4(4.0f, 0.0f, 0.0f, 1.0f));
    inputVertices.push_back(make_float4(5.0f, 4.0f, 0.0f, 1.0f));
    inputVertices.push_back(make_float4(3.0f, 3.0f, 0.0f, 1.0f));

    inputVertices.push_back(make_float4(-3.5f, 1.0f, 0.0f, 1.0f));
    inputVertices.push_back(make_float4(-4.0f, -1.0f, 0.0f, 1.0f));
    inputVertices.push_back(make_float4(-3.0f, -1.0f, 0.0f, 1.0f));

    inputVertices.push_back(make_float4(2.0f, -2.0f, 0.0f, 1.0f));
    inputVertices.push_back(make_float4(4.0f, -3.0f, 0.0f, 1.0f));
    inputVertices.push_back(make_float4(3.5f, -1.0f, 0.0f, 1.0f));

    for (unsigned int i=0; i<inputVertices.size(); i++) inputColors.push_back(make_float4(0.0f, 0.0f, 1.0f, 1.0f));
    for (unsigned int i=0; i<inputVertices.size(); i++) inputNormals.push_back(make_float3(0.0f, 0.0f, 1.0f));


#ifdef RTI_EXAMPLE
    inputVertices.assign(isosurface2->vertices_begin(), isosurface2->vertices_end());
    inputNormals.assign(isosurface2->normals_begin(), isosurface2->normals_end());
    inputColors.assign(thrust::make_transform_iterator(isosurface2->scalars_begin(), color_map<float>(31.0f, 500.0f)),
                       thrust::make_transform_iterator(isosurface2->scalars_end(), color_map<float>(31.0f, 500.0f)));
#endif

#ifdef TANGLE_EXAMPLE
    inputVertices.assign(isosurface->vertices_begin(), isosurface->vertices_end());
    inputNormals.assign(isosurface->normals_begin(), isosurface->normals_end());
    inputColors.assign(thrust::make_transform_iterator(isosurface->scalars_begin(), color_map<float>(31.0f, 500.0f)),
                       thrust::make_transform_iterator(isosurface->scalars_end(), color_map<float>(31.0f, 500.0f)));
#endif

    renders = new render<thrust::device_vector<float4>::iterator, thrust::device_vector<float3>::iterator, thrust::device_vector<float4>::iterator>(inputVertices.begin(),
                   inputNormals.begin(), inputColors.begin(), inputVertices.size(), viewportWidth, viewportHeight);

    zoomLevelBase = cameraFOV = 40.0; cameraZ = 2.0; zoomLevelPct = zoomLevelPctDefault = 0.5;
    center_pos = make_float3(0, 0, 0);
    cameraFOV = zoomLevelBase*zoomLevelPct;  camera_up = make_float3(0,1,0);

    inputVerticesHost = inputVertices;
    inputNormalsHost = inputNormals;
    inputColorsHost = inputColors;

    return 0;
}
