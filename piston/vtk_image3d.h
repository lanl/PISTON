/*
 * vtk_image3d.h
 *
 *  Created on: Oct 25, 2011
 *      Author: ollie
 */

#ifndef VTK_IMAGE3D_H_
#define VTK_IMAGE3D_H_

#include <vtkImageData.h>

namespace piston {

template <typename IndexType, typename ValueType, typename Space>
struct vtk_image3d : public piston::image3d<int, ValueType, Space>
{
    typedef piston::image3d<IndexType, ValueType, Space> Parent;

    typedef typename detail::choose_container<typename Parent::CountingIterator, thrust::tuple<IndexType, IndexType, IndexType> >::type GridCoordinatesContainer;
    GridCoordinatesContainer grid_coordinates_vector;
    typedef typename GridCoordinatesContainer::iterator GridCoordinatesIterator;

    typedef typename detail::choose_container<typename Parent::CountingIterator, ValueType>::type PointDataContainer;
    PointDataContainer point_data_vector;
    typedef typename PointDataContainer::iterator PointDataIterator;

    vtk_image3d(vtkImageData *image) :
	Parent(image->GetDimensions()[0], image->GetDimensions()[1], image->GetDimensions()[2]),
	grid_coordinates_vector(Parent::grid_coordinates_begin(), Parent::grid_coordinates_end()),
	point_data_vector((ValueType *) image->GetScalarPointer(),
	                  (ValueType *) image->GetScalarPointer() + this->NPoints) {}

    void resize(int xdim, int ydim, int zdim) {
 	Parent::resize(xdim, ydim, zdim);
 	// TBD, is there resize in VTK?
     }

     GridCoordinatesIterator grid_coordinates_begin() {
 	return grid_coordinates_vector.begin();
     }
     GridCoordinatesIterator grid_coordinates_end() {
 	return grid_coordinates_vector.end();
     }

     PointDataIterator point_data_begin() {
 	return point_data_vector.begin();
     }
     PointDataIterator point_data_end() {
 	return point_data_vector.end();
     }
};

}
#endif /* VTK_IMAGE3D_H_ */
