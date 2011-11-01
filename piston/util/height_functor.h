/*
 * height_functor.h
 *
 *  Created on: Oct 21, 2011
 *      Author: ollie
 */

#ifndef HEIGHT_FUNCTOR_H_
#define HEIGHT_FUNCTOR_H_

#include <piston/implicit_function.h>

namespace piston
{

template <typename IndexType, typename ValueType>
struct height_functor : public piston::implicit_function3d<IndexType, ValueType>
{
    typedef piston::implicit_function3d<IndexType, ValueType> Parent;
    typedef typename Parent::InputType InputType;

    __host__ __device__
    ValueType operator()(InputType pos) const {
	return thrust::get<2>(pos);
    };
};

} // namespace piston


#endif /* HEIGHT_FUNCTOR_H_ */
