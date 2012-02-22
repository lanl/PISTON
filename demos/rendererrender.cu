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

#define GLYPH_BUFFER_SIZE 12000000
int rcnt = 0;

RendererRender::RendererRender()
{
    mouse_buttons = 0;
    translate = make_float3(0.0, 0.0, 0.0);
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
    if (rcnt == 0) (*(renders))();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0*viewportWidth, 0.0, 1.0*viewportHeight, 0.1, 100.0); //gluPerspective(cameraFOV, 2.0, 0.01, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0);
    glPushMatrix();

    //qrot.getRotMat(rotationMatrix);
    //glMultMatrixf(rotationMatrix);

    glColor4f(0.5, 0.5, 0.5, 1.0);

    /*
    glColor4f(0.5, 0.5, 0.5, 1.0);
    glEnableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    glNormalPointer(GL_FLOAT, 0, &inputNormalsHost[0]);
    //glColorPointer(4, GL_FLOAT, 0, &inputColorsHost[0]);
    glVertexPointer(3, GL_FLOAT, 0, &inputVerticesHost[0]);
    glDrawElements(GL_TRIANGLES, 3*inputIndicesHost.size(), GL_UNSIGNED_INT, &inputIndicesHost[0]);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);*/

    glPopMatrix();

    if (rcnt == 0) screenShot("test.tga", viewportWidth, viewportHeight, true);
    rcnt++;
}


void RendererRender::screenShot(std::string fileName, unsigned int width, unsigned int height, bool includeAlpha)
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

    /*char* pBuffer = new char[pixelSize*width*height ];

    std::cout << "Size: " << pixelSize << std::endl;

    for (unsigned int i=0; i<width; i++)
    {
      for (unsigned int j=0; j<height; j++)
      {
        pBuffer[i*height*pixelSize + j*pixelSize + 0] = 0;
        pBuffer[i*height*pixelSize + j*pixelSize + 1] = 0;
        pBuffer[i*height*pixelSize + j*pixelSize + 2] = 255;
        if (includeAlpha) pBuffer[i*height*pixelSize + j*pixelSize + 3] = 255;
      }
    }*/

    //glReadPixels( 0,0,width,height,pixelFormat,GL_UNSIGNED_BYTE,pBuffer );

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

    //delete [] pBuffer;
}


void RendererRender::cleanup()
{

}


void RendererRender::initGL(bool aAllowInterop)
{
    viewportWidth = 128;  viewportHeight = 128;
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

    //printf("Error code: %s\n", cudaGetErrorString(errorCode));
    read();
}


int RendererRender::read()
{
    inputVertices.push_back(make_float3(10.0f, 10.0f, 0.0f));
    inputVertices.push_back(make_float3(50.0f, 11.0f, 0.0f));
    inputVertices.push_back(make_float3(11.0f, 50.0f, 0.0f));

    std::cout << "Viewport size: " << viewportWidth << " " << viewportHeight << std::endl;

    renders = new render<thrust::device_vector<float3>::iterator>(inputVertices.begin(), inputVertices.size(), viewportWidth, viewportHeight);

    zoomLevelBase = cameraFOV = 40.0; cameraZ = 2.0; zoomLevelPct = zoomLevelPctDefault = 0.5;
    center_pos = make_float3(0, 0, 0);
    cameraFOV = zoomLevelBase*zoomLevelPct;  camera_up = make_float3(0,1,0);

    inputVerticesHost = inputVertices;

    return 0;
}
