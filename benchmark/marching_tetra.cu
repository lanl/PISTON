/*
 * marching_tetra.cu
 *
 *  Created on: Sep 4, 2012
 *      Author: ollie
 */

#include <sys/time.h>

#include <vtkImageData.h>
#include <vtkRTAnalyticSource.h>

#include <piston/vtk_image3d.h>
#include "piston/image3d_to_tetrahedrons.h"
#include "piston/marching_tetrahedron.h"


//#define SPACE thrust::host_space_tag
#define SPACE thrust::detail::default_device_space_tag

using namespace piston;

int
main()
{
    vtkRTAnalyticSource *src = vtkRTAnalyticSource::New();
    src->SetWholeExtent(-100, 100, -100, 100, -100, 100);
    src->Update();

    vtk_image3d<SPACE> image(src->GetOutput());

    // get max and min of 3D scalars
    float min_iso = *thrust::min_element(image.point_data_begin(), image.point_data_end());
    float max_iso = *thrust::max_element(image.point_data_begin(), image.point_data_end());

    typedef image3d_to_tetrahedrons<vtk_image3d<SPACE> > tetra_source;
    tetra_source tetra(image);

    marching_tetrahedron<tetra_source, tetra_source> isosurface(tetra, tetra, 160.0f);

    struct timeval begin, end, diff;
    gettimeofday(&begin, 0);
    for (float isovalue = min_iso; isovalue < max_iso; isovalue += ((max_iso-min_iso)/50)) {
	isosurface.set_isovalue(isovalue);
	isosurface();
    }
    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);

    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    std::cout << "total time: " << seconds << ", fps: " << 50.f/seconds << std::endl;
    return 0;

}
