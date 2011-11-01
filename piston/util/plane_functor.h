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

    const float3 origin;
    const float3 normal;

    plane_functor(float3 origin, float3 normal) :
	origin(origin), normal(normal) {}

    __host__ __device__
    float operator()(int pointId) const {
	const IndexType x = thrust::get<0>(pos);
	const IndexType y = thrust::get<1>(pos);
	const IndexType z = thrust::get<2>(pos);
        return dot(make_float3(x, y, z) - origin, normal);
    }
};

}

#endif /* PLANE_FUNCTOR_H_ */
