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

#include <vtkFloatArray.h>
#include <vtkRectilinearGrid.h>
#include <vtkRectilinearGridGeometryFilter.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkImageViewer2.h>
#include <vtkContourFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkActor.h>
#include <vtkActor2D.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkCamera.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkXMLImageDataReader.h>
#include <vtkScalarBarActor.h>
#include <vtkLookupTable.h>
#include <sys/time.h>
#include <unistd.h>
#include <GL/glut.h>
#include <sstream>

int dataSetIndex, numIters, numFrames, saveFrames;
float maxIso, minIso, isoIter, deltaIso; 
float isovalue, delta;
char frameDirectory[1024];
float center_pos[3];
float offset[3] = {0.0, 0.0, 0.0};
float cameraFOV = 18.0;

#define PACKED __attribute__((packed))

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

struct Rect
{
        int left,top,right,bottom;
};


struct TGAHeader
{
        unsigned char  identsize                ;   // size of ID field that follows 18 uint8 header (0 usually)
        unsigned char  colourmaptype            ;   // type of colour map 0=none, 1=has palette
        unsigned char  imagetype                ;   // type of image 0=none,1=indexed,2=rgb,3=grey,+8=rle packed

        unsigned short colourmapstart   PACKED;   // first colour map entry in palette
        unsigned short colourmaplength  PACKED;   // number of colours in palette
        unsigned char  colourmapbits          ;   // number of bits per palette entry 15,16,24,32

        unsigned short xstart                   PACKED;   // image x origin
        unsigned short ystart                   PACKED;   // image y origin
        unsigned short width                    PACKED;   // image width in pixels
        unsigned short height                   PACKED;   // image height in pixels
        unsigned char  bits                           ;   // image bits per pixel 8,16,24,32
        unsigned char  descriptor                     ;   // image descriptor bits (vh flip bits)

        inline bool IsFlippedHorizontal() const
        {
                return (descriptor & 0x10) != 0;
        }

        inline bool IsFlippedVertical() const
        {
                return (descriptor & 0x20) != 0;
        }

        // pixel data follows header
};


void ScreenShot( std::string fileName, unsigned int width, unsigned int height, bool includeAlpha = false )
{
        unsigned int pixelSize = 3;
        unsigned int pixelSizeBits = 24;
        GLenum pixelFormat = GL_BGR_EXT;

        if(includeAlpha)
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


float* ColorMap(float ival, float imin, float imax)
{
  float colorSaturation = 0.6;
  float pct = (ival-imin)/(imax-imin);
  float B = colorSaturation*(1.0 - pct);
  float R = colorSaturation*(pct);
  float G = colorSaturation*(0.5 - fabs(pct - 0.5));
  float* cmap = new float[3];
  cmap[0] = R; cmap[1] = G; cmap[2] = B;
  return cmap; 
}


int main(int argc, char **argv)
{
    vtkXMLImageDataReader *reader;
    reader = vtkXMLImageDataReader::New();

    if (argc < 2)
    {
      std::cout << "Usage: vtiContourVTK dataSetIndex numIters numFrames framesDirectory" << std::endl;
      return 1;
    }
    dataSetIndex = atoi(argv[1]);
    numIters = 0;
    numFrames = 600;
    saveFrames = 0;
    if (argc > 2) numIters = atoi(argv[2]);
    if (argc > 3) numFrames = atoi(argv[3]);
    if (argc > 4) { saveFrames = 1; sprintf(frameDirectory, "%s", argv[4]); }

    char metafile[1024]; char fname[1024]; char dtag[1024];
    fname[0] = 0; float isovaluePct, cameraZ, zoomLevelPctDefault;
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
        if (tag.compare("isovalues") == 0)  sscanf(line.c_str(), "%s %f %f %f", dtag, &minIso, &maxIso, &isovaluePct);
        if (tag.compare("zoom") == 0)       sscanf(line.c_str(), "%s %f %f %f", dtag, &cameraFOV, &cameraZ, &zoomLevelPctDefault);
      }
      myfile.close();
    }
    else
    {
    	cout << "Unable to open file";
    	return 0;
    }

    char filename[1024];
    sprintf(filename, "%s/%s", STRINGIZE_VALUE_OF(DATA_DIRECTORY), fname);

    reader->SetFileName(filename);
    reader->Update();

    center_pos[0] = 431.0; center_pos[1] = 450.0; center_pos[2] =  95.0;
    isoIter = 0.1;
    isovalue = minIso;
    deltaIso = (maxIso-minIso)/(isoIter*numFrames);
    delta = deltaIso;

    // Create five surfaces F(x,y,z) = constant between range specified
    vtkContourFilter *contours = vtkContourFilter::New();
    contours->SetInput(reader->GetOutput());

    // map the contours to graphical primitives
    vtkPolyDataMapper *contMapper = vtkPolyDataMapper::New();
    contMapper->SetInput(contours->GetOutput());
    contMapper->ScalarVisibilityOff();

    // create an actor for the contours
    vtkActor *contActor = vtkActor::New();
    contActor->SetMapper(contMapper);
    contActor->GetProperty()->SetColor(1.0, 0.0, 0.0);

    // Create the usual rendering stuff.
    vtkRenderer *renderer = vtkRenderer::New();
    vtkRenderWindow *renWin = vtkRenderWindow::New();

    renWin->SetSize(2048, 1024);

    renWin->AddRenderer(renderer);
    vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
    iren->SetRenderWindow(renWin);

    renderer->AddActor(contActor);
    renderer->SetBackground(1,1,1);
    renderer->ResetCamera();
    renderer->SetBackground(0.0, 0.0, 0.0);

    struct timeval begin, end, diff;
    float seconds, curFPS;
    double timerTotal = 0.0;
    int timerCount = 0;
    if (numIters > 0)
    {
      float avg = 0.0;
      for (int i=0; i<numIters; i++)
      {
    	gettimeofday(&end, 0);
    	timersub(&end, &begin, &diff);
    	seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    	curFPS = 1.0/seconds;
    	std::cout << "Total fps: " << curFPS << std::endl;
    	timerCount++;
    	if (timerCount > 50)
    	{
    	  timerTotal += 1.0/seconds;
    	  std::cout << "Averages: " << timerCount << ": " << timerTotal/(1.0*timerCount-50.0) << std::endl;
    	}
    	gettimeofday(&begin, 0);

    	isovalue = minIso + 0.0*(maxIso-minIso);
    	float value = isovalue;
    	value += (rand() % 100)/100.0;

        std::cout << "Generating isovalue " << value << std::endl;

        contours->SetValue(0, value);
        contours->Update();
      }
      
      return 0;
    }

    vtkScalarBarActor *colorBar = vtkScalarBarActor::New();
    vtkLookupTable *lookup = vtkLookupTable::New();
    lookup->SetNumberOfColors(64);
    lookup->SetHueRange(0.0, 0.667);
    lookup->Build();
    for (int i=0; i<64; i++)
    {
      float* colors = ColorMap(minIso + (i/64.0)*(maxIso-minIso), minIso, maxIso);
      lookup->SetTableValue(i, colors[0], colors[1], colors[2]);
    }
    colorBar->SetLookupTable(lookup); 
    colorBar->SetTitle("Density (normalized)");
    colorBar->SetHeight(0.25);
    colorBar->SetWidth(0.1);
    colorBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    colorBar->GetPositionCoordinate()->SetValue(0.02, 0.75);
    renderer->AddActor(colorBar);

    contActor->SetOrigin(center_pos[0], center_pos[1], center_pos[2]);
    contActor->RotateZ(90.0);
  
    contours->SetValue(0, minIso+0.001*(maxIso-minIso));
    contours->Update();
    float* colors = ColorMap(minIso, minIso, maxIso);
    contActor->GetProperty()->SetColor(colors[0], colors[1], colors[2]);

    renWin->Render();
    renderer->GetActiveCamera()->SetViewAngle(cameraFOV);
    contActor->SetPosition(offset[0], offset[1], offset[2]);
    renderer->GetActiveCamera()->SetClippingRange(200.0, 4000.0);
    renWin->Render();

    // interact with data
    int frameCount = 0;
    while (frameCount <= numFrames) 
    {
        struct timeval begin, end, diff;
        gettimeofday(&begin, 0); 

	    contours->SetValue(0, isovalue);
        contours->Update();

        gettimeofday(&end, 0);
        timersub(&end, &begin, &diff);

        float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
        std::cout << "time seconds: " << seconds << " fps: " << 1.0/seconds << std::endl;

        float* colors = ColorMap(isovalue, minIso, maxIso);
        contActor->GetProperty()->SetColor(colors[0], colors[1], colors[2]);

        renWin->Render();

        if ((saveFrames) && (frameCount < numFrames))
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
          ScreenShot(screenShotFile,glRect.right - glRect.left,glRect.bottom - glRect.top,false);
          std::cout << "Output frame " << frameCount << std::endl;
        }

        isovalue += delta;
        if (isovalue > maxIso)
        {
          delta = -deltaIso;
          isovalue = maxIso;
        }
        if (isovalue < minIso)
        {
	      delta = deltaIso;
          isovalue = minIso;
        }

        contActor->RotateY(360.0/numFrames);
        frameCount++;
  }
    
  return 0;
}
