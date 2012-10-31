/*
 * vthTetra.cpp
 *
 *  Created on: Sep 6, 2012
 *      Author: ollie
 */

#include <vtkImageData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkRTAnalyticSource.h>
#include <vtkDataSetTriangleFilter.h>
#include <vtkContourFilter.h>

#include <algorithm>
#include <sys/time.h>

int
main(int argc, char *argv[])
{
    int grid_size = atoi(argv[1])/2;

    vtkRTAnalyticSource *src = vtkRTAnalyticSource::New();
    src->SetWholeExtent(-grid_size, grid_size, -grid_size, grid_size, -grid_size, grid_size);
    src->Update();

    vtkImageData *image = src->GetOutput();

    float min_iso = *std::min_element((float *) image->GetScalarPointer(),
                                      (float *) image->GetScalarPointer() + image->GetNumberOfPoints());

    float max_iso = *std::max_element((float *) image->GetScalarPointer(),
                                      (float *) image->GetScalarPointer() + image->GetNumberOfPoints());

    vtkDataSetTriangleFilter *tetra = vtkDataSetTriangleFilter::New();
    tetra->SetInput(src->GetOutput());
    tetra->Update();

    vtkContourFilter *contour = vtkContourFilter::New();
    contour->SetInput(tetra->GetOutput());
    contour->GenerateValues(50, min_iso, max_iso);

    struct timeval begin, end, diff;
    gettimeofday(&begin, 0);

    contour->Update();

    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);

    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    std::cout << "grid_size: " << grid_size*2 << ", total time: " << seconds << ", fps: " << 50.0f/seconds << std::endl;
    return 0;
}


