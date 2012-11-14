#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/iterator/constant_iterator.h>

#include <iostream>
#include <typeinfo>
#include <mpi.h>

namespace dthrust
{

// Shut down MPI cleanly if something goes wrong
/*void my_abort(int err)
{
    std::cout << "Test FAILED\n";
    MPI_Abort(MPI_COMM_WORLD, err);
}*/

// Error handling macros
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
        std::cerr << "MPI error calling \""#call"\"\n"; \
        MPI_Abort(MPI_COMM_WORLD, -1); }

template <typename T>
void get_mpi_type(const std::type_info& t, MPI_Datatype& dt, int& df);


template <typename T>
void host_to_device(int hsize, thrust::host_vector<T>& h, thrust::device_vector<T>& d);


template <typename T>
void device_to_host(int hsize, thrust::device_vector<T>& d, thrust::host_vector<T>& h);


template <typename InputIterator, typename OutputIterator, typename T, typename BinaryOperation>
OutputIterator scan(InputIterator first, InputIterator last, OutputIterator result, T init, bool inclusiveScan, BinaryOperation binop);


template <typename InputIterator, typename OutputIterator, typename BinaryOperation>
OutputIterator inclusive_scan(InputIterator first, InputIterator last, OutputIterator result, BinaryOperation binop)
{
    return dthrust::scan(first, last, result, 0, true, binop);
}


template <typename InputIterator, typename OutputIterator, typename T, typename BinaryOperation>
OutputIterator exclusive_scan(InputIterator first, InputIterator last, OutputIterator result, T init, BinaryOperation binop)
{
    return dthrust::scan(first, last, result, init, false, binop);
}


template<typename InputIterator1 , typename InputIterator2 , typename OutputIterator , typename BinaryFunction >
OutputIterator transform(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, BinaryFunction op)
{
    return thrust::transform(first1, last1, first2, result, op);
}


template<typename InputIterator1 , typename OutputIterator , typename BinaryFunction >
OutputIterator transform(InputIterator1 first1, InputIterator1 last1, OutputIterator result, BinaryFunction op)
{
    return thrust::transform(first1, last1, result, op);
}


template<typename InputIterator, typename OutputVector>
typename OutputVector::iterator upper_bound_counting(InputIterator first, InputIterator last, int cntMax, OutputVector& result); 


template <typename T>
void output_global_vector(thrust::device_vector<T>& testing, int gsize, int lsize);

}

#include "dthrust.inl"

//#endif
