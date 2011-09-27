/*
 * sphere.h
 *
 *  Created on: Sep 20, 2011
 *      Author: ollie
 */

#ifndef SPHERE_H_
#define SPHERE_H_

#include <piston/implicit_function.h>

namespace piston
{

template <typename IndexType, typename ValueType>
struct sphere : piston::implicit_function3d<IndexType, ValueType>
{
    typedef piston::implicit_function3d<IndexType, ValueType> Parent;
    typedef typename Parent::InputType InputType;

    const IndexType x_o;
    const IndexType y_o;
    const IndexType z_o;
    const IndexType radius;

    sphere(IndexType x, IndexType y, IndexType z, IndexType radius) :
	x_o(x), y_o(y), z_o(z), radius(radius) {}

    __host__ __device__
    ValueType operator()(InputType pos) const {
	const IndexType x = thrust::get<0>(pos);
	const IndexType y = thrust::get<1>(pos);
	const IndexType z = thrust::get<2>(pos);
	const IndexType xx = x - x_o;
	const IndexType yy = y - y_o;
	const IndexType zz = z - z_o;
	return (xx*xx + yy*yy + zz*zz); //- radius*radius);
    }
};

} // namespace piston
#endif /* SPHERE_H_ */
