/*
 * plane_functor.h
 *
 *  Created on: Oct 24, 2011
 *      Author: ollie
 */

#ifndef PLANE_FUNCTOR_H_
#define PLANE_FUNCTOR_H_

#include <piston/implicit_function.h>

namespace piston
{

template <typename IndexType, typename ValueType>
struct plane_functor : public piston::implicit_function3d<IndexType, ValueType>
{
    typedef piston::implicit_function3d<IndexType, ValueType> Parent;
    typedef typename Parent::InputType InputType;

    int pointsPerLayer, xDim, yDim, zDim;
    const float3 origin;
    const float3 normal;

    plane_functor(float3 origin, float3 normal, int xDim, int yDim, int zDim) :
	origin(origin), normal(normal), xDim(xDim), yDim(yDim), zDim(zDim), pointsPerLayer(yDim*xDim) {}

    __host__ __device__
    float operator()(InputType pos) const {
    	const IndexType xc = thrust::get<0>(pos);
    	const IndexType yc = thrust::get<1>(pos);
    	const IndexType zc = thrust::get<2>(pos);

    	// scale and shift such that x, y, z <- [-1,1]
    	const float x = 2*static_cast<float>(xc)/(xDim-1) - 1;
    	const float y = 2*static_cast<float>(yc)/(yDim-1) - 1;
    	const float z = 2*static_cast<float>(zc)/(zDim-1) - 1;

        return dot(make_float3(x, y, z) - origin, normal);
    }
};

}

#endif /* PLANE_FUNCTOR_H_ */
