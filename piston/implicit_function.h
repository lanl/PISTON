/*
 * implicit_function.h
 *
 *  Created on: Aug 23, 2011
 *      Author: ollie
 */

#ifndef IMPLICIT_FUNCTION_H_
#define IMPLICIT_FUNCTION_H_

#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace piston
{
    template <typename IndexType, typename ValueType>
    struct implicit_function2d : public thrust::unary_function<thrust::tuple<IndexType, IndexType>, ValueType>
    {
	typedef thrust::tuple<IndexType, IndexType> InputType;
    };

    template <typename IndexType, typename ValueType>
    struct implicit_function3d : public thrust::unary_function<thrust::tuple<IndexType, IndexType, IndexType>, ValueType>
    {
	typedef thrust::tuple<IndexType, IndexType, IndexType> InputType;
    };
}

#endif /* IMPLICIT_FUNCTION_H_ */
