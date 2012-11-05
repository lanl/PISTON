/*
 * marching_cube.cu
 *
 *  Created on: Sep 7, 2012
 *      Author: ollie
 */

#include <sys/time.h>

#include <vtkImageData.h>
#include <vtkRTAnalyticSource.h>

#include <piston/vtk_image3d.h>
#include "piston/marching_cube.h"

using namespace piston;

int
main(int argc, char *argv[])
{
    int grid_size = 128;
    if (argc > 1)
	grid_size = atoi(argv[1])/2;

    vtkRTAnalyticSource *src = vtkRTAnalyticSource::New();
    src->SetWholeExtent(-grid_size, grid_size, -grid_size, grid_size,
                        -grid_size, grid_size);
    src->Update();

    vtk_image3d<> image(src->GetOutput());

    // get max and min of 3D scalars
    float min_iso = *thrust::min_element(image.point_data_begin(),
                                         image.point_data_end());
    float max_iso = *thrust::max_element(image.point_data_begin(),
                                         image.point_data_end());

    typedef vtk_image3d<> image_source;
    marching_cube<image_source> isosurface(image, image);

    struct timeval begin, end, diff;
    gettimeofday(&begin, 0);
    for (float isovalue = min_iso; isovalue < max_iso;
	 isovalue += ((max_iso-min_iso)/50)) {
	isosurface.set_isovalue(isovalue);
	isosurface();
    }
    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);

    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    std::cout << "grid_size: " << grid_size*2
	      << ", total time: " << seconds
	      << ", fps: " << 50.f/seconds << std::endl;
    return 0;

}



