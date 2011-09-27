/*
 * choose_container.inl
 *
 *  Created on: Jan 25, 2011
 *      Author: ollie
 */

#ifndef CHOOSE_CONTAINER_INL_
#define CHOOSE_CONTAINER_INL_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace piston
{

namespace detail {

template<typename Iterator,
	 typename ValueType = typename thrust::iterator_traits<Iterator>::value_type,
	 typename Space = typename thrust::iterator_space<Iterator>::type>
struct choose_container;

template<typename Iterator, typename ValueType>
struct choose_container<Iterator, ValueType, thrust::detail::default_device_space_tag>
{
    typedef thrust::device_vector<ValueType> type;
};

template<typename Iterator, typename ValueType>
struct choose_container<Iterator, ValueType, thrust::host_space_tag>
{
    typedef thrust::host_vector<ValueType> type;
};

} // namespace detail

} // namespace piston

#endif /* CHOOSE_CONTAINER_INL_ */
