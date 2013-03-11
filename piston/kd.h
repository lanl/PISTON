/*
Copyright (c) 2012, Los Alamos National Security, LLC
All rights reserved.
Copyright 2012. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL),
which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.

NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.

If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
·         Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
·         Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other
          materials provided with the distribution.
·         Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used
          to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Christopher Sewell, csewell@lanl.gov
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <thrust/merge.h>
#include<thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <math.h>
#include <sys/time.h>

#include <iostream>

// When TEST is defined, this example will use a small fixed input data set and output all results
// When TEST is not defined, this example will use a large random data set and output only timings
//#define TEST

// When TEST is not defined, use this data size for the randomized example input
#define INPUT_SIZE 24474 //524288

// Typedefs for convenience
typedef typename thrust::counting_iterator<int>	CountingIterator;


//===========================================================================
/*!    
    \class      KDTree

    \brief      
    Constructs a KD-tree for given 3D input points using a data-parallel 
    algorithm
*/
//===========================================================================
class KDTree
{
  public:

    //==========================================================================
    /*! 
        struct multiply

        Multiply the vector elements by a constant factor
    */
    //==========================================================================
    struct multiply : public thrust::unary_function<int, int>
    {
        float value;

        __host__ __device__
        multiply(float value) : value(value) { };

        __host__ __device__
        int operator() (int i)        
	{
          return ((int)(value*i));
        }
    };


    //==========================================================================
    /*! 
        struct medianSplit

        Return whether the element's rank is greater than or equal to the 
        median rank for its segment (in order to set flags for a split)
    */
    //==========================================================================
    struct medianSplit : public thrust::unary_function<int, int>
    {
        int *segmentMedianRanks, *ranks;

        __host__ __device__
        medianSplit(int* segmentMedianRanks, int* ranks) : segmentMedianRanks(segmentMedianRanks), ranks(ranks) { };

        __host__ __device__
        int operator() (int i)
        {
          return (ranks[i] >= segmentMedianRanks[i]);
        }
    };


    //==========================================================================
    /*! 
        struct calculateIndex

        Compute the index to which this element should be permuted in a split as
        follows:  if its flag is false, its index should be the sum of all 
        elements in previous segments plus the number of false flags so far
        in its segment (excluding itself);  if its flag is true, its index should
        be the sum of all elements in previous segments plus the total number of
        false flags in the entire current segment plus the number of true flags
        so far in its segment (excluding itself)
    */
    //==========================================================================
    struct calculateIndex : public thrust::unary_function<int, int>
    {
        int *flags, *curSegTrueCountEx, *totalPrevSegs, *curSegFalseCountIn, *segEndInd;

        __host__ __device__
        calculateIndex(int* flags, int* curSegTrueCountEx, int* totalPrevSegs, int* curSegFalseCountIn, int* segEndInd) : 
              flags(flags), curSegTrueCountEx(curSegTrueCountEx), totalPrevSegs(totalPrevSegs), curSegFalseCountIn(curSegFalseCountIn), segEndInd(segEndInd) { };

        __host__ __device__
        int operator() (int i)
        {
          if (flags[i] == 0) return (totalPrevSegs[i] + curSegFalseCountIn[i] - 1);
          return (totalPrevSegs[i] + curSegFalseCountIn[segEndInd[i]-1] + curSegTrueCountEx[i]);
        }
    };


    //==========================================================================
    /*! 
        struct newKeys

        Compute the new segment id for the element as two times its previous 
        segment id, plus one if its flag is true
    */
    //==========================================================================
    struct newKeys : public thrust::unary_function<int, int>
    {
        int *segmentIds, *flags;

        __host__ __device__
        newKeys(int* segmentIds, int* flags) : segmentIds(segmentIds), flags(flags) { };

        __host__ __device__
        int operator() (int i)
        {
          return (2*segmentIds[i]+flags[i]);
        }
    };


    //==========================================================================
    /*! 
        struct inverse

        Return 1 for a 0, and 0 for a 1
    */
    //==========================================================================
    struct inverse : public thrust::unary_function<int,int>
    {
        __host__ __device__
        int operator()(int x) const
        {
          return (1-x);
        }
    };


    //==========================================================================
    /*! 
        Member variable declarations
    */
    //==========================================================================

    //! Flags and segment ids
    thrust::device_vector<int> m_flags, m_segmentIds;

    //! Vectors containing x, y, and z ranks for all nodes at each level of the tree 
    std::vector<thrust::device_vector<int>*> m_xranks, m_yranks, m_zranks;

    //! Vectors containing x, y, and z ranks for all nodes at current tree level
    thrust::device_vector<int> m_xrank, m_yrank, m_zrank;

    //! Vector containing point ids at each level of the tree
    std::vector<thrust::device_vector<int>*> m_pointIds;

    //! Vector containing point ids at current tree level
    thrust::device_vector<int> m_pointId;

    //! Vectors for intermediate steps of algorithms that can be reused in different methods 
    thrust::device_vector<int> m_temp1, m_temp2, m_temp3, m_temp4, m_temp5;
    thrust::device_vector<float> m_tempf1; 

    //! Vectors for intermediate steps of algorithms that are to be used in only one method
    thrust::device_vector<int> m_tempA, m_tempB, m_tempC, m_tempD, m_tempE;  

    //! Store the ranks at each level?
    bool m_saveAllLevels;

    //! Total number of levels for the tree (tree height)
    int m_maxLevel;


    //==========================================================================
    /*! 
        Constructor for KDTree class

        \fn	KDTree::KDTree
    */
    //==========================================================================
    KDTree() : m_saveAllLevels(false) {};


    ~KDTree()
    {
        m_flags.clear();  m_segmentIds.clear();
        m_xrank.clear();  m_yrank.clear();  m_zrank.clear();
        m_pointId.clear();
        m_temp1.clear();  m_temp2.clear();  m_temp3.clear();  m_temp4.clear();  m_temp5.clear();
        m_tempf1.clear();
        m_tempA.clear();  m_tempB.clear();  m_tempC.clear();  m_tempD.clear();  m_tempE.clear();
    }


    //==========================================================================
    /*! 
        Perform a segmented split operation on the input values based on the 
        input flags and segment ids

        \fn	KDTree::segmentedSplit
	\param	a_values
	\param	a_flags
	\param	a_segmentIds
    */
    //==========================================================================
    void segmentedSplit(std::vector<thrust::device_vector<int>::iterator> a_values, thrust::device_vector<int>& a_flags, thrust::device_vector<int>& a_segmentIds)
    {
        // Get data size
        int n = a_flags.size();

        // Construct a vector that contains for each element the total number of true flags preceding it within its segment
        thrust::exclusive_scan_by_key(a_segmentIds.begin(), a_segmentIds.end(), a_flags.begin(), m_temp4.begin());    

        // Construct a vector that contains for each element the total number of elements in previous segments
        thrust::inclusive_scan_by_key(a_segmentIds.begin(), a_segmentIds.end(), CountingIterator(0), m_temp2.begin(), thrust::equal_to<int>(), thrust::minimum<int>());  

        // Construct a vector that contains for each element the index of the last element in its segment (actually, the index of the first element in the next segment)
        thrust::inclusive_scan_by_key(a_segmentIds.rbegin(), a_segmentIds.rend(), thrust::make_reverse_iterator(thrust::make_counting_iterator(n+1)), m_temp1.rbegin(), thrust::equal_to<int>(), thrust::maximum<int>());

        // Construct a vector that contains for each element the total number of false flags up to and including it within its segment
        thrust::inclusive_scan_by_key(a_segmentIds.begin(), a_segmentIds.end(), thrust::make_transform_iterator(a_flags.begin(), inverse()), m_temp3.begin());

        // Construct a vector that contains for each element the index to which its value should be permuted in the split
        thrust::transform(CountingIterator(0), CountingIterator(0)+n, m_temp5.begin(), calculateIndex(thrust::raw_pointer_cast(&*a_flags.begin()),   // flags
                                                                                                      thrust::raw_pointer_cast(&*m_temp4.begin()),   // exclusive running total of true flags in current segment
                                                                                                      thrust::raw_pointer_cast(&*m_temp2.begin()),   // total number of elements in previous segments
                                                                                                      thrust::raw_pointer_cast(&*m_temp3.begin()),   // inclusive running total of false flags in current segment
                                                                                                      thrust::raw_pointer_cast(&*m_temp1.begin()))); // indexes of last element in current segment

        // Split the point ids, flags, and segment ids based on the computed indices, and copy the results of the split back to the original vectors (OpenMP apparently can't do the scatters in-place)
        for (unsigned int i=0; i<a_values.size(); i++)
        { 
          thrust::scatter(a_values[i], a_values[i]+n, m_temp5.begin(), m_temp1.begin());
          thrust::copy(m_temp1.begin(), m_temp1.begin()+n, a_values[i]);
        }
    }


    //==========================================================================
    /*! 
        Convert the ranks such that the elements of each segment have ranks 0
        through s-1 (where s is the number of elements in the segment), 
        maintaining the relative ordering of the ranks from the input (e.g., if
        the input ranks for a segment are 5 2 9 1, the output ranks for that
        segment should be 2 1 3 0)

        \fn	KDTree::renumberRanks
	\param	a_ranks
	\param	a_offsets
	\param	a_flags
	\param	a_segmentIds
	\param  a_originalSegmentIds
    */
    //==========================================================================
    void renumberRanks(thrust::device_vector<int>& a_ranks, thrust::device_vector<int>& a_offsets, thrust::device_vector<int>& a_flags, 
                       thrust::device_vector<int>& a_segmentIds, thrust::device_vector<int>& a_originalSegmentIds)
    {
        // Get data size
        int n = a_ranks.size();

        // Add the total number of elements in previous segments to the (within-segment) ranks
        thrust::transform(a_ranks.begin(), a_ranks.end(), a_offsets.begin(), m_temp4.begin(), thrust::plus<int>());

        // Construct a segmented counting iterator (starting at zero in each segment)
        thrust::exclusive_scan_by_key(a_segmentIds.begin(), a_segmentIds.end(), thrust::make_constant_iterator(1), m_tempA.begin());

        // Scatter the counting iterator values and flags based on the ranks
        thrust::scatter(m_tempA.begin(), m_tempA.end(), m_temp4.begin(), m_tempB.begin());
        thrust::scatter(a_flags.begin(), a_flags.end(), m_temp4.begin(), m_tempC.begin());

        // Split the scattered counting iterator values based on the flags
        std::vector<thrust::device_vector<int>::iterator> itemsToSplit;
        itemsToSplit.push_back(m_tempB.begin());  itemsToSplit.push_back(m_tempC.begin());
        segmentedSplit(itemsToSplit, m_tempC, a_originalSegmentIds);

        // Construct a vector that contains for each element the total number of elements in previous segments
        thrust::inclusive_scan_by_key(a_segmentIds.begin(), a_segmentIds.end(), CountingIterator(0), m_temp3.begin(), thrust::equal_to<int>(), thrust::minimum<int>());

        // Add the total number of elements in previous segments to the (within-segment) ranks
        thrust::transform(m_tempB.begin(), m_tempB.end(), m_temp3.begin(), m_tempB.begin(), thrust::plus<int>());

        // Scatter the counting iterator values based on the ranks
        thrust::scatter(m_tempA.begin(), m_tempA.end(), m_tempB.begin(), a_ranks.begin());
    }


    //==========================================================================
    /*! 
        Perform a segmented rank split operation on the input point ids and x, y,
        and z ranks, based on the input flags and segment ids, by making calls to 
        the segmentedSplit and renumberRanks methods 

        \fn	KDTree::segmentedRankSplit
	\param	a_pointIds
	\param	a_xranks
        \param  a_yranks
        \param  a_zranks
	\param	a_flags
	\param	a_segmentIds
    */
    //==========================================================================
    void segmentedRankSplit(thrust::device_vector<int>& a_pointIds, thrust::device_vector<int>& a_xranks, thrust::device_vector<int>& a_yranks, thrust::device_vector<int>& a_zranks, 
                            thrust::device_vector<int>& a_flags, thrust::device_vector<int>& a_segmentIds)
    {
        // Get data size
        int n = a_flags.size();

        // Back up the original (pre-split) segment ids
        thrust::copy(a_segmentIds.begin(), a_segmentIds.end(), m_tempE.begin());

        // Construct a vector that contains for each element the total number of elements in  previous segments
        thrust::inclusive_scan_by_key(a_segmentIds.begin(), a_segmentIds.end(), CountingIterator(0), m_tempD.begin(), thrust::equal_to<int>(), thrust::minimum<int>());
 
        // Split the point ids, ranks, and flags based on the flags, within each segment
        std::vector<thrust::device_vector<int>::iterator> itemsToSplit;
        itemsToSplit.push_back(a_pointIds.begin());  itemsToSplit.push_back(a_xranks.begin());  itemsToSplit.push_back(a_yranks.begin());  itemsToSplit.push_back(a_zranks.begin());  itemsToSplit.push_back(a_flags.begin());
        segmentedSplit(itemsToSplit, a_flags, a_segmentIds);
 
        // Compute the new segment id for each element as two times its previous segment id, plus one if its flag is true
        thrust::transform(CountingIterator(0), CountingIterator(0)+n, a_segmentIds.begin(), newKeys(thrust::raw_pointer_cast(&*a_segmentIds.begin()), thrust::raw_pointer_cast(&*a_flags.begin())));
 
        // Within each segment, renumber the x, y, and z ranks from zero to the segment size minus one, maintaining the relative ordering
        renumberRanks(a_xranks, m_tempD, a_flags, a_segmentIds, m_tempE);
        renumberRanks(a_yranks, m_tempD, a_flags, a_segmentIds, m_tempE);
        renumberRanks(a_zranks, m_tempD, a_flags, a_segmentIds, m_tempE);
    }


    //==========================================================================
    /*! 
        Compute global ranks by sorting input coordinates

        \fn	KDTree::computeGlobalRanks
	\param	a_values
	\param	a_ranks
    */
    //==========================================================================
    void computeGlobalRanks(thrust::device_vector<float>& a_values, thrust::device_vector<int>& a_ranks)
    {
        // Get data size
        int n = a_values.size();

        // Create a counting iterator (actually in memory, since it will get reordered by the sort)
        thrust::copy(CountingIterator(0), CountingIterator(0)+n, m_temp1.begin());

        // If in test mode, print out the input coordinates
        #ifdef TEST
          thrust::copy(a_values.begin(), a_values.end(), std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
        #endif

        // Sort a key-value pair of the coordinate values and indexes, first copying the values to prevent reordering the input data itself
        thrust::copy(a_values.begin(), a_values.end(), m_tempf1.begin());
        thrust::sort_by_key(m_tempf1.begin(), m_tempf1.end(), m_temp1.begin());

        // Scatter counting iterator values by the output of the sort to get a vector with the rank of each element 
        thrust::scatter(CountingIterator(0), CountingIterator(0)+n, m_temp1.begin(), a_ranks.begin());

        // If in test mode, print out the computed ranks
        #ifdef TEST
          thrust::copy(a_ranks.begin(), a_ranks.end(), std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
        #endif
    } 


    //==========================================================================
    /*! 
        Compute flags for the given rank vector by comparing each element to the 
        median rank for its segment

        \fn	KDTree::computeFlags
	\param	a_segmentIds
	\param	a_XRanks
        \param  a_flags
    */
    //==========================================================================
    void computeFlags(thrust::device_vector<int>& a_segmentIds, thrust::device_vector<int>& a_ranks, thrust::device_vector<int>& a_flags)
    {
        // Get data size
        int n = a_segmentIds.size();

        // Construct a segmented counting iterator (starting at one in each segment)
        thrust::inclusive_scan_by_key(a_segmentIds.begin(), a_segmentIds.end(), thrust::make_constant_iterator(1), m_temp1.begin());

        // Construct a vector that contains for each element the total number of elements in its segment
        thrust::inclusive_scan_by_key(a_segmentIds.rbegin(), a_segmentIds.rend(), m_temp1.rbegin(), m_temp2.rbegin(), thrust::equal_to<int>(), thrust::maximum<int>());
        
        // Construct a vector that contains for each element the median rank index in its segment (equal to half the total number of elements in its segment)
        thrust::transform(m_temp2.begin(), m_temp2.end(), m_temp2.begin(), multiply(0.5));

        // Set the flag for each element depending on whether its rank is greater than or equal to the median rank for its segment
        thrust::transform(CountingIterator(0), CountingIterator(0)+n, a_flags.begin(), medianSplit(thrust::raw_pointer_cast(&*m_temp2.begin()), thrust::raw_pointer_cast(&*a_ranks.begin())));
    }


    //==========================================================================
    /*! 
        Output the values from device vectors at a tree level for debugging

        \fn	KDTree::outputValues
        \param  a_level
	\param	a_pointIds
	\param	a_XRank
        \param  a_YRank
        \param  a_ZRank
	\param	a_flags
	\param	a_segmentIds
    */
    //==========================================================================
    void outputValues(int a_level, thrust::device_vector<int>& a_pointIds, thrust::device_vector<int>& a_xranks, thrust::device_vector<int>& a_yranks, thrust::device_vector<int>& a_zranks, 
                      thrust::device_vector<int>& a_flags, thrust::device_vector<int>& a_segmentIds)
    {
          int n = a_pointIds.size();  if (n > 10) n = 10;
          std::cout << "Segmented Split Output Level " << a_level << ": " << std::endl;
          std::cout << "Point ids   "; thrust::copy(a_pointIds.begin(), a_pointIds.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
          std::cout << "XRanks      "; thrust::copy(a_xranks.begin(), a_xranks.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
          std::cout << "YRanks      "; thrust::copy(a_yranks.begin(), a_yranks.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
          std::cout << "ZRanks      "; thrust::copy(a_zranks.begin(), a_zranks.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
          std::cout << "Flags       "; thrust::copy(a_flags.begin(), a_flags.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
          std::cout << "Segment ids "; thrust::copy(a_segmentIds.begin(), a_segmentIds.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
    }


    //==========================================================================
    /*! 
        Create a level of the KD tree

        \fn	KDTree::createTree
	\param	a_X
	\param	a_Y
        \param  a_Z    
    */
    //==========================================================================
    void buildTreeLevel(int level)
    {
        // Perform a segmented rank split operation
        segmentedRankSplit(m_pointId, m_xrank, m_yrank, m_zrank, m_flags, m_segmentIds);
  
        // Save the computed ranks and point ids from this tree level
        if (m_saveAllLevels)
        {
          thrust::copy(m_xrank.begin(), m_xrank.end(), m_xranks[level]->begin()); 
          thrust::copy(m_yrank.begin(), m_yrank.end(), m_yranks[level]->begin());
          thrust::copy(m_zrank.begin(), m_zrank.end(), m_zranks[level]->begin());
          thrust::copy(m_pointId.begin(), m_pointId.end(), m_pointIds[level]->begin());
        }

        // If in test mode, print out the computed point ids, ranks, flags, and segment ids for this tree level
        #ifdef TEST
          outputValues(level, m_pointId, m_xrank, m_yrank, m_zrank, m_flags, m_segmentIds);
        #endif
       
        // Alternate splitting along the x, y, and z dimensions
        if ((level+1) < m_maxLevel)
        {
          if ((level-1) % 3 == 0) computeFlags(m_segmentIds, m_yrank, m_flags);
          if ((level-1) % 3 == 1) computeFlags(m_segmentIds, m_zrank, m_flags);
          if ((level-1) % 3 == 2) computeFlags(m_segmentIds, m_xrank, m_flags);
        }
    }


    //==========================================================================
    /*! 
        Create all levels of the KD tree

        \fn	KDTree::createTree
	\param	a_X
	\param	a_Y
        \param  a_Z    
    */
    //==========================================================================    
    void buildFullTree()
    {
        // Start timing
        struct timeval begin, end, diff;
        gettimeofday(&begin, 0);

        // Construct all levels of the KD tree
        for (unsigned int level=1; level<m_maxLevel; level++)
        {
          // Print out the current level being computed
            std::cout << "Level " << level << std::endl;

          // Build the tree for this level
          buildTreeLevel(level);
        }

        // Print out the time taken for the KD-tree construction
        gettimeofday(&end, 0);
        timersub(&end, &begin, &diff);
        float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
          std::cout << "Time elapsed: " << seconds << " s"<< std::endl;
    }


    //==========================================================================
    /*! 
        Initialize a KD tree

        \fn	KDTree::createTree
	\param	a_X
	\param	a_Y
        \param  a_Z    
    */
    //==========================================================================
    void initializeTree(thrust::device_vector<float>& a_x, thrust::device_vector<float>& a_y, thrust::device_vector<float>& a_z)
    {
        // Get data size
        int n = a_x.size();

        // Allocate memory for all class member variables based on the input size
        m_flags.resize(n); m_segmentIds.resize(n);       
        m_temp1.resize(n); m_temp2.resize(n); m_temp3.resize(n); m_temp4.resize(n); m_temp5.resize(n); m_tempf1.resize(n);
        m_tempA.resize(n); m_tempB.resize(n); m_tempC.resize(n); m_tempD.resize(n); m_tempE.resize(n); 

        // Compute the number of levels the tree will have based on the input size
        m_maxLevel = ceil(log(n)/log(2) + 1); 

        // Allocate memory to store the point ids and ranks in each dimension at each tree level
        if (m_saveAllLevels)
        {
          m_xranks.resize(m_maxLevel);
          m_yranks.resize(m_maxLevel);
          m_zranks.resize(m_maxLevel);
          m_pointIds.resize(m_maxLevel);
          for (unsigned int level=0; level<m_maxLevel; level++)
          {
            m_xranks[level] = new thrust::device_vector<int>();  
            m_xranks[level]->resize(n);  
            m_yranks[level] = new thrust::device_vector<int>();  
            m_yranks[level]->resize(n); 
            m_zranks[level] = new thrust::device_vector<int>();  
            m_zranks[level]->resize(n); 
            m_pointIds[level] = new thrust::device_vector<int>();  
            m_pointIds[level]->resize(n);
          }
        }
        m_xrank.resize(n);  m_yrank.resize(n);  m_zrank.resize(n);  m_pointId.resize(n);

        // Initialize the point ids at the top level as consecutive integers
        thrust::fill(m_segmentIds.begin(), m_segmentIds.end(), 0);
        thrust::copy(CountingIterator(0), CountingIterator(0)+n, m_pointId.begin());

        // Compute the global ranks in each dimension based on the input coordinate values
        computeGlobalRanks(a_x, m_xrank);       
        computeGlobalRanks(a_y, m_yrank);  
        computeGlobalRanks(a_z, m_zrank);     
  
        // Compute the flags for the first split based on the ranks in the x dimension
        computeFlags(m_segmentIds, m_xrank, m_flags);

        // Copy the computed ranks and point ids from this level to initialize their values at the next tree level
        if (m_saveAllLevels)
        {
          thrust::copy(m_xrank.begin(), m_xrank.end(), m_xranks[0]->begin()); 
          thrust::copy(m_yrank.begin(), m_yrank.end(), m_yranks[0]->begin());
          thrust::copy(m_zrank.begin(), m_zrank.end(), m_zranks[0]->begin());
          thrust::copy(m_pointId.begin(), m_pointId.end(), m_pointIds[0]->begin());
        }
 
        // If in test mode, print out the point ids, ranks, flags, and segment ids for the first tree level
        #ifdef TEST    
          outputValues(0, m_pointId, m_xrank, m_yrank, m_zrank, m_flags, m_segmentIds);
        #endif
    }
};





