/*
 * vthTetra.cpp
 *
 *  Created on: Sep 6, 2012
 *      Author: ollie
 */

#include <vtkImageData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkImageMandelbrotSource.h>
#include <vtkRTAnalyticSource.h>
#include <vtkDataSetTriangleFilter.h>
#include <vtkDataSetToUnstructuredGridFilter.h>
#include <vtkContourFilter.h>

#include <algorithm>
#include <sys/time.h>

int
main()
{
    vtkRTAnalyticSource *src = vtkRTAnalyticSource::New();
    src->SetWholeExtent(-100, 100, -100, 100, -100, 100);
    src->Update();

    vtkImageData *image = src->GetOutput();

    float min_iso = *std::min_element((float *) image->GetScalarPointer(),
                                      (float *) image->GetScalarPointer() + image->GetNumberOfPoints());

    float max_iso = *std::max_element((float *) image->GetScalarPointer(),
                                      (float *) image->GetScalarPointer() + image->GetNumberOfPoints());

    std::cout << "min_iso: " << min_iso << std::endl;
    std::cout << "max_iso: " << max_iso << std::endl;

    vtkDataSetTriangleFilter *tetra = vtkDataSetTriangleFilter::New();
    tetra->SetInput(src->GetOutput());

    vtkContourFilter *contour = vtkContourFilter::New();
    contour->SetInput(tetra->GetOutput());
//    contour->Update();
    contour->GenerateValues(50, min_iso, max_iso);

    struct timeval begin, end, diff;
    gettimeofday(&begin, 0);
//    for (float isovalue = min_iso; isovalue < max_iso; isovalue += ((max_iso-min_iso)/50)) {
//	contour->SetValue(0, isovalue);
//	contour->Update();
//    }
    contour->Update();
    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);

    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    std::cout << "total time: " << seconds << ", fps: " << 50.f/seconds << std::endl;
    return 0;
}


