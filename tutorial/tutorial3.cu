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

// When TEST is defined, this example will use a small fixed input data set and output all results
// When TEST is not defined, this example will use a large random data set and output only timings
#define TEST

// When TEST is not defined, use this data size for the randomized example input
#define INPUT_SIZE 24474 

#include "kd.h"


//==========================================================================
/*! 
    struct randomInit

    Initialize the vector elements with random values
*/
//==========================================================================
struct randomInit : public thrust::unary_function<float, float>
{
    __host__ __device__
    randomInit() { };

    __host__ __device__
    float operator() (float i)
    {
      return ((rand() % 1000000)/100000.0);
    }
};


//==========================================================================
/*! 
    Entry point for the example program that uses the KDTree class

    \fn	main
*/
//==========================================================================
int main(void)
{
    // Declare host and device vectors for the input coordinates for this example
    thrust::device_vector<float> X, Y, Z;
    thrust::host_vector<float> XH, YH, ZH;

    // If in test mode, initialize the coordinates with hard-coded values
    #ifdef TEST
      int n = 8; X.resize(n); Y.resize(n); Z.resize(n);
      X[0] = 2.89383; X[1] = 8.85386; X[2] = 2.38335; X[3] = 6.36915; X[4] = 9.30886; X[5] = 6.92777; X[6] = 7.47793; X[7] = 7.60492;
      Y[0] = 2.02362; Y[1] = 5.20059; Y[2] = 4.90027; Y[3] = 5.13926; Y[4] = 8.97763; Y[5] = 6.41421; Y[6] = 5.16649; Y[7] = 3.6869;
      Z[0] = 3.83426; Z[1] = 7.02567; Z[2] = 5.95368; Z[3] = 1.8054;  Z[4] = 0.89172; Z[5] = 0.05211; Z[6] = 9.56429; Z[7] = 4.55736;

    // Otherwise, create large input arrays with random values; compute in host space and then copy to device space since the rand function does not exist in CUDA
    #else
      int n = INPUT_SIZE; X.resize(n); Y.resize(n); Z.resize(n); XH.resize(n); YH.resize(n); ZH.resize(n);
      thrust::transform(XH.begin(), XH.end(), XH.begin(), randomInit());
      thrust::transform(YH.begin(), YH.end(), YH.begin(), randomInit());
      thrust::transform(ZH.begin(), ZH.end(), ZH.begin(), randomInit());
      X = XH; Y = YH; Z = ZH;
    #endif

    // Create an instance of the KDTree class, and use it to construct a KD tree with the given input coordinates
    KDTree tree;
    tree.initializeTree(X, Y, Z);
    tree.buildFullTree();
    return 0;
}

