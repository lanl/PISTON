/*
 *  Copyright 2008-2011 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/device/dereference.h>

#include <vector>
#include <numeric>
#include "omp.h"

namespace thrust
{
namespace detail
{
namespace device
{
namespace omp
{

// Scan function based on Belloch's algorithm
template <class InputIterator, class OutputIterator, class BinaryOperation>
OutputIterator scan(InputIterator first, InputIterator last, OutputIterator result, bool inclusiveScan, BinaryOperation binop)
{
  // If there is only one processor or one or fewer data elements, don't do extra work
  int numThreads = omp_get_max_threads();
  int N = last - first;
  if (N <= 0) return (result);
  if (N == 1) { if (inclusiveScan) result[0] = first[0]; else result[0] = 0; return (result + N); } 
  if (numThreads < 2)
  {
    if (inclusiveScan) return std::partial_sum(first, last, result, binop);

    typename std::iterator_traits<InputIterator>::value_type lastItem = first[0];
    result[0] = 0;
    typename std::iterator_traits<InputIterator>::value_type newLastItem;
    typename std::iterator_traits<InputIterator>::value_type prevResult = 0;
    for (int i=1; i<N; i++)
    {
      newLastItem = first[i];
      prevResult = result[i] = binop(prevResult, lastItem);
      lastItem = newLastItem;
    }
    return (result + N);
  }

  // Initialize variables for data size and how many elements each processor gets
  if (numThreads > N) numThreads = N;
  int itemsPerThread = N / numThreads;

  // Each processor sums (with respect to binary scan operator) all elements assigned to it
  std::vector<typename std::iterator_traits<InputIterator>::value_type> processorSums(numThreads); 
  std::vector<typename std::iterator_traits<InputIterator>::value_type> processorSuppl(numThreads-1);
  std::fill(processorSuppl.begin(), processorSuppl.end(), 0);
  int sumItemsPerThread = std::max(1, ((numThreads-1)*itemsPerThread)/numThreads);
  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    if (id < numThreads - 1)
    {
      int startIndex = id*itemsPerThread;
      int endIndex = startIndex + sumItemsPerThread;
      typename std::iterator_traits<InputIterator>::value_type sum = first[startIndex];
      for (int i=startIndex+1; i<endIndex; i++)
        sum = binop(sum, first[i]);
      processorSums[id] = sum;
    }
    else if (id == numThreads - 1)
    {
      for (int s=0; s<numThreads-1; s++)
      {
        int startIndex = s*itemsPerThread + sumItemsPerThread;
        int endIndex = (s+1)*itemsPerThread;
        typename std::iterator_traits<InputIterator>::value_type psum = first[startIndex];
        for (int i=startIndex+1; i<endIndex; i++)
          psum = binop(psum, first[i]);
        if (endIndex > startIndex) 
          processorSuppl[s] = psum;
      }
    }
  }
  
  for (int i=0; i<numThreads-1; i++)
    processorSums[i] = binop(processorSums[i], processorSuppl[i]);
  
  // Perform a scan across the processor sums to get offsets for each processor
  typename std::iterator_traits<InputIterator>::value_type lastItem = processorSums[0];
  processorSums[0] = 0;
  typename std::iterator_traits<InputIterator>::value_type newLastItem = processorSums[1];
  processorSums[1] = lastItem;
  lastItem = newLastItem;
  for (int i=2; i<numThreads; i++)
  {
    typename std::iterator_traits<InputIterator>::value_type newLastItem = processorSums[i]; 
    processorSums[i] = binop(processorSums[i-1], lastItem);
    lastItem = newLastItem;
  }
 
  // Each processor scans the elements assigned to it, using result of processor scan above as offset
  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    if (id < numThreads)
    {
      typename std::iterator_traits<InputIterator>::value_type lastItem = first[id*itemsPerThread]; 
      if (inclusiveScan) result[id*itemsPerThread] = binop(processorSums[id], lastItem);
      else result[id*itemsPerThread] = processorSums[id];
      int firstIndex = id*itemsPerThread+1;
      int lastIndex = (id+1)*itemsPerThread;
      if (id == numThreads-1) lastIndex = N;

      typename std::iterator_traits<InputIterator>::value_type newLastItem;
      typename std::iterator_traits<InputIterator>::value_type prevResult = result[firstIndex-1];
      if (inclusiveScan)
      {
        for (int i=firstIndex; i<lastIndex; i++)
          prevResult = result[i] = binop(prevResult, first[i]);
      }
      else
      {
        for (int i=firstIndex; i<lastIndex; i++)
        {
          newLastItem = first[i]; 
          prevResult = result[i] = binop(prevResult, lastItem); 
          lastItem = newLastItem;
        }
      }
    }
  }
  return (result + N);
}

template<typename InputIterator,
         typename OutputIterator,
         typename AssociativeOperator>
  OutputIterator inclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                AssociativeOperator binary_op)
{
    return scan(first, last, result, true, binary_op);
}

template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename AssociativeOperator>
  OutputIterator exclusive_scan(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                AssociativeOperator binary_op)
{
   return scan(first, last, result, false, binary_op);
}

} // end namespace omp
} // end namespace device
} // end namespace detail
} // end namespace thrust

