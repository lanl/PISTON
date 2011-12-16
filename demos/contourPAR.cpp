/*
Copyright (c) 2011, Los Alamos National Security, LLC
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
    	and/or other materials provided with the distribution.
    Neither the name of the Los Alamos National Laboratory nor the names of its contributors may be used to endorse or promote products derived from this
    	software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "vtkActor.h"
#include "vtkAppendPolyData.h"
#include "vtkCamera.h"
#include "vtkConeSource.h"
#include "vtkContourFilter.h"
#include "vtkDataSet.h"
#include "vtkElevationFilter.h"
#include "vtkImageReader.h"
#include "vtkMath.h"
#include "vtkMPIController.h"
#include "vtkParallelFactory.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkTestUtilities.h"
#include "vtkRegressionTestImage.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkWindowToImageFilter.h"
#include "vtkImageData.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformation.h"
#include "vtkXMLImageDataReader.h"

#include "vtkDebugLeaks.h"

#include <sstream>
#include <sys/time.h>
#include <mpi.h>

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

// Just pick a tag which is available
static const int ISO_VALUE_RMI_TAG=300; 
static const int ISO_OUTPUT_TAG=301;

int dataSetIndex, numIters;
float maxIso, minIso, deltaIso; 
float isovalue;

struct ParallelIsoArgs_tmp
{
  int* retVal;
  int argc;
  char** argv;
};

struct ParallelIsoRMIArgs_tmp
{
  vtkContourFilter* ContourFilter;
  vtkMultiProcessController* Controller;
  vtkElevationFilter* Elevation;
};

// call back to set the iso surface value.
void SetIsoValueRMI(void *localArg, void* vtkNotUsed(remoteArg), 
                    int vtkNotUsed(remoteArgLen), int vtkNotUsed(id))
{ 
  ParallelIsoRMIArgs_tmp* args = (ParallelIsoRMIArgs_tmp*)localArg;

  float val;

  vtkContourFilter *iso = args->ContourFilter;
  val = iso->GetValue(0);
  iso->SetValue(0, val + deltaIso);
  args->Elevation->Update();

  vtkMultiProcessController* contrl = args->Controller;
  contrl->Send(args->Elevation->GetOutput(), 0, ISO_OUTPUT_TAG);
}


// This will be called by all processes
void MyMain( vtkMultiProcessController *controller, void *arg )
{
  vtkXMLImageDataReader *reader;
  vtkContourFilter *iso;
  vtkElevationFilter *elev;
  int myid, numProcs;
  float val;
  ParallelIsoArgs_tmp* args = reinterpret_cast<ParallelIsoArgs_tmp*>(arg);
  
  // Obtain the id of the running process and the total
  // number of processes
  myid = controller->GetLocalProcessId();
  numProcs = controller->GetNumberOfProcesses();

  // Create the reader
  reader = vtkXMLImageDataReader::New();

  char metafile[1024]; char fname[1024]; char dtag[1024];
  fname[0] = 0; float isovaluePct; //, cameraZ, zoomLevelPctDefault;
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
    }
    myfile.close();
  }
  else
  {
    cout << "Unable to open file";
    return;
  }

  char filename[1024];
  sprintf(filename, "%s/%s", STRINGIZE_VALUE_OF(DATA_DIRECTORY), fname);

  reader->SetFileName(filename);
  reader->Update();

  int isoLoops = numIters;
  if (numIters == 0)
  {
    isoLoops = 1;
    deltaIso = (maxIso - minIso)/(1000.0);
    minIso = minIso + (maxIso - minIso) / 2.0;
  } 
  else
  {
    deltaIso = (maxIso - minIso)/(1.0*numIters);
  }

  // Iso-surface.
  iso = vtkContourFilter::New();
  iso->SetInputConnection(reader->GetOutputPort());
  iso->SetValue(0, minIso);
  iso->ComputeScalarsOff();
  iso->ComputeGradientsOff();
  
  // Compute a different color for each process.
  elev = vtkElevationFilter::New();
  elev->SetInputConnection(iso->GetOutputPort());
  val = (myid+1) / static_cast<float>(numProcs);
  elev->SetScalarRange(val, val+0.001);

  // Tell the pipeline which piece we want to update.
  vtkStreamingDemandDrivenPipeline* exec = 
    vtkStreamingDemandDrivenPipeline::SafeDownCast(elev->GetExecutive());
  exec->SetUpdateNumberOfPieces(exec->GetOutputInformation(0), numProcs);
  exec->SetUpdatePiece(exec->GetOutputInformation(0), myid);

  if (myid != 0)
  {
    // If I am not the root process
    ParallelIsoRMIArgs_tmp args2;
    args2.ContourFilter = iso;
    args2.Controller = controller;
    args2.Elevation = elev;

    // Last, set up a RMI call back to change the iso surface value.
    // This is done so that the root process can let this process
    // know that it wants the contour value to change.
    controller->AddRMI(SetIsoValueRMI, (void *)&args2, ISO_VALUE_RMI_TAG);
    controller->ProcessRMIs();
  }
  else
  {
    // Create the rendering part of the pipeline
    vtkAppendPolyData *app = vtkAppendPolyData::New();
    vtkRenderer *ren = vtkRenderer::New();
    vtkRenderWindow *renWindow = vtkRenderWindow::New();
    vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
    vtkPolyDataMapper *mapper = vtkPolyDataMapper::New();
    vtkActor *actor = vtkActor::New();
    //vtkCamera *cam = vtkCamera::New();
    renWindow->AddRenderer(ren);
    iren->SetRenderWindow(renWindow);
    ren->SetBackground(0.0, 0.0, 0.0);
    renWindow->SetSize(2048, 1024);
    mapper->SetInputConnection(app->GetOutputPort());
    actor->SetMapper(mapper);
    ren->AddActor(actor);

    struct timeval begin, end, diff;
    gettimeofday(&begin, 0); 

    // loop through some iso surface values.
    for (int j = 0; j < isoLoops; ++j)
    {
      // set the local value
      isovalue = minIso + 0.0*(maxIso-minIso);
      float value = isovalue;
      value += (rand() % 100)/100.0;

      iso->SetValue(0, value); //iso->GetValue(0) + deltaIso);
      elev->Update();

      for (int i = 1; i < numProcs; ++i)
      {
        // trigger the RMI to change the iso surface value.
        controller->TriggerRMI(i, ISO_VALUE_RMI_TAG);      
      }
      for (int i = 1; i < numProcs; ++i)
      {
        vtkPolyData* pd = vtkPolyData::New();
        controller->Receive(pd, i, ISO_OUTPUT_TAG);
        if (j == isoLoops - 1)
        {
          app->AddInput(pd);
        }
        pd->Delete();
      }
    }

    // Tell the other processors to stop processing RMIs.
    for (int i = 1; i < numProcs; ++i)
    {
      controller->TriggerRMI(i, vtkMultiProcessController::BREAK_RMI_TAG); 
    }

    vtkPolyData* outputCopy = vtkPolyData::New();
    outputCopy->ShallowCopy(elev->GetOutput());
    app->AddInput(outputCopy);
    outputCopy->Delete();
    app->Update();
    
    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);
    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    std::cout << " fps: " << (1.0*isoLoops)/seconds << std::endl;

    if (numIters == 0)
    {
      renWindow->Render(); 
      iren->Start();
    }

    // Clean up
    app->Delete();
    ren->Delete();
    renWindow->Delete();
    iren->Delete();
    mapper->Delete();
    actor->Delete();
    //cam->Delete();
    }
  
  // clean up objects in all processes.
  reader->Delete();
  iso->Delete();
  elev->Delete();
}


int main( int argc, char* argv[] )
{
  // Command-line parameters
  if (argc < 2)
  {
    std::cout << "Usage: vtiContourPAR dataSetIndex numIters" << std::endl;
    return 1;
  }
  dataSetIndex = atoi(argv[1]);
  if (argc < 3) numIters = 0;
  else numIters = atoi(argv[2]);

  // This is here to avoid false leak messages from vtkDebugLeaks when
  // using mpich. It appears that the root process which spawns all the
  // main processes waits in MPI_Init() and calls exit() when
  // the others are done, causing apparent memory leaks for any objects
  // created before MPI_Init().
  MPI_Init(&argc, &argv);

  // Note that this will create a vtkMPIController if MPI
  // is configured, vtkThreadedController otherwise.
  vtkMPIController* controller = vtkMPIController::New();

  controller->Initialize(&argc, &argv, 1);

  vtkParallelFactory* pf = vtkParallelFactory::New();
  vtkObjectFactory::RegisterFactory(pf);
  pf->Delete();
 
  // Added for regression test.
  // ----------------------------------------------
  int retVal = 1;
  ParallelIsoArgs_tmp args;
  args.retVal = &retVal;
  args.argc = argc;
  args.argv = argv;
  // ----------------------------------------------

  controller->SetSingleMethod(MyMain, &args);
  controller->SingleMethodExecute();

  controller->Finalize();
  controller->Delete();

  return !retVal;
}





