/*
Copyright (c) 2011, Los Alamos National Security, LLC
All rights reserved.
Copyright 2011. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL),
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
*/

#ifndef RENDER_H_
#define RENDER_H_

#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <thrust/merge.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <piston/image3d.h>
#include <piston/piston_math.h>
#include <piston/choose_container.h>

#include <float.h>
#include <math.h>
#include <queue>

//#define SCANLINE
#define USE_KD_TREE

namespace piston {

class DisplayInfo
{
public:
    float cameraFOV, zNear, zFar;
    int left, right, bottom, top;

    float3 lightAmb, lightDif;
    float kc, kl, kq;
    float4 lp;

    DisplayInfo()
    {
      cameraFOV = 20.0f;  zNear = -2000.0f;  zFar = 2000.0f;
      left = 0;  right = 256;  bottom = 0;  top = 256;

      lightAmb = make_float3(0.5f, 0.5f, 0.5f);
      lightDif = make_float3(0.5f, 0.5f, 0.5f);
      kc = 1.0f;  kl = 0.0f;  kq = 0.0f;
      lp = make_float4(0.0f, 0.0f, 10000.0f, 1.0f);
    }
};


struct TreeNode
{
    int level;
    int splitDim;
    float splitVal;
    int startIndex;
    int numTris;
};

template <typename InputVertices>
class KdTree
{
public:
    InputVertices inputVertices;
    int nTriangles;
    std::vector<TreeNode> htree;
    thrust::device_vector<TreeNode> tree;
    thrust::device_vector<int> S;
    thrust::device_vector<int> T;
    thrust::device_vector<int> P;
    int levels;

    KdTree() {}

    void buildTree(InputVertices aInputVertices, int aNTriangles, int aLevels)
    {
	inputVertices = aInputVertices;
	nTriangles = aNTriangles;
	levels = aLevels;  int treeSize = 1;

	for (unsigned int i=0; i<levels; i++) treeSize *= 2;
	htree.resize(treeSize);
	tree.resize(treeSize);
	S.resize(2*nTriangles);
	T.resize(2*nTriangles);
	P.resize(3*nTriangles);
	thrust::sequence(S.begin(), S.begin()+nTriangles);
	thrust::fill(S.begin()+nTriangles, S.end(), -1);

	std::cout << "Sizes: " << nTriangles << " " << 3*nTriangles << " " << levels << std::endl;

	htree[0].level = 0;  htree[0].startIndex = 0;  htree[0].numTris = nTriangles;

	for (unsigned int l=0; l<levels-1; l++)
	{
	  thrust::fill(T.begin(), T.end(), -1);
	  int numNodes = 1;  for (unsigned int i=0; i<l; i++) numNodes *= 2;
	  int offset = 0;
	  for (unsigned int n=0; n<numNodes; n++)
	  {
	    int curIndex = numNodes + n;

	    float4 csums = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	    for (unsigned int j=0; j<3; j++)
	      csums = csums + thrust::reduce(//inputVertices, inputVertices+htree[curIndex-1].numTris, make_float4(0,0,0,0), sum4());
		        thrust::make_permutation_iterator(inputVertices, make_transform_iterator(S.begin()+htree[curIndex-1].startIndex, saxpy(3,j))),
	    	        thrust::make_permutation_iterator(inputVertices, make_transform_iterator(S.begin()+htree[curIndex-1].startIndex, saxpy(3,j))) + htree[curIndex-1].numTris,
	    	        make_float4(0,0,0,0), sum4());

	    htree[curIndex-1].splitDim = l % 2;
	    if (htree[curIndex-1].numTris == 0) htree[curIndex-1].splitVal = 0.0f;
	    else htree[curIndex-1].splitVal = htree[curIndex-1].splitDim == 0 ? csums.x / (3.0f*htree[curIndex-1].numTris) : csums.y / (3.0f*htree[curIndex-1].numTris);

	    if (offset >= T.size())
	    {
	      levels = l;
	      std::cout << "Returning early with " << levels << " levels" << std::endl;
	      thrust::copy(P.begin(), P.end(), S.begin());
	      thrust::copy(htree.begin(), htree.begin()+treeSize, tree.begin());
	      return;
	    }

	    thrust::copy_if(S.begin()+htree[curIndex-1].startIndex, S.begin()+htree[curIndex-1].startIndex+htree[curIndex-1].numTris, T.begin()+offset, splitLine(inputVertices, htree[curIndex-1].splitDim, htree[curIndex-1].splitVal, 0));
	    int nsplit1 = thrust::count_if(T.begin()+offset, T.end(), validIndex());
	    thrust::copy_if(S.begin()+htree[curIndex-1].startIndex, S.begin()+htree[curIndex-1].startIndex+htree[curIndex-1].numTris, T.begin()+offset+nsplit1, splitLine(inputVertices, htree[curIndex-1].splitDim, htree[curIndex-1].splitVal, 1));
	    int nsplit2 = thrust::count_if(T.begin()+offset+nsplit1, T.end(), validIndex());

	    htree[2*curIndex-1].level = l + 1;
	    htree[2*curIndex-1].startIndex = offset;
	    htree[2*curIndex-1].numTris = nsplit1;
	    offset += nsplit1;

	    htree[2*curIndex].level = l + 1;
	    htree[2*curIndex].startIndex = offset;
	    htree[2*curIndex].numTris = nsplit2;
	    offset += nsplit2;
	  }
	  thrust::copy(S.begin(), S.end(), P.begin());
	  thrust::copy(T.begin(), T.end(), S.begin());
	}

	thrust::copy(htree.begin(), htree.end(), tree.begin());

	//std::cout << S[rand() % 140] << " " << htree[rand() % 140].startIndex << std::endl;
	/*std::cout << "Output: " << std::endl;
	for (unsigned int i=0; i<3*nTriangles; i++)
	  std::cout << S[i] << " ";
	std::cout << std::endl;*/

	/*std::cout << "Tree nodes: " << std::endl;
	for (unsigned int i=0; i<treeSize-1; i++)
	  std::cout << "Node " << i+1 << ": " << htree[i].startIndex << " " << htree[i].numTris << " " << htree[i].level << " " << htree[i].splitDim << " " << htree[i].splitVal << std::endl;
	std::cout << std::endl;

        std::cout << "Finished building tree" << std::endl;*/
    }

    struct saxpy : public thrust::unary_function<int, int>
    {
	int m, a;

        __host__ __device__
        saxpy(int m, int a) : m(m), a(a) { };

        __host__ __device__
        int operator() (int i)
        {
          return (m*i+a);
        }
    };

    struct sum4 : public thrust::binary_function<float4, float4, float4>
    {
    	__host__ __device__
    	sum4() { };

    	__host__ __device__
    	float4 operator() (float4 c1, float4 c2)
    	{
    	  return make_float4(c1.x+c2.x, c1.y+c2.y, c1.z+c2.z, c1.w+c2.w);
    	}
    };

    struct splitLine
    {
	InputVertices inputVertices;
	int dim, side;
	float val;

	__host__ __device__
	splitLine(InputVertices inputVertices, int dim, float val, int side) : inputVertices(inputVertices), dim(dim), val(val), side(side) { };

    	__host__ __device__
    	bool operator() (int i)
    	{
    	  float4 vt0 = *(inputVertices+3*i+0);
    	  float4 vt1 = *(inputVertices+3*i+1);
    	  float4 vt2 = *(inputVertices+3*i+2);
    	  bool v0, v1, v2;
    	  if (dim == 0) { v0 = vt0.x < val;  v1 = vt1.x < val;  v2 = vt2.x < val; }
    	  if (dim == 1) { v0 = vt0.y < val;  v1 = vt1.y < val;  v2 = vt2.y < val; }

    	  if (side == 0) { return (v0 || v1 || v2); }
    	  else { return (!v0 || !v1 || !v2); }
    	}
    };

    struct validIndex
    {
	__host__ __device__
        validIndex() { };

	__host__ __device__
	bool operator() (int i)
	{
	  return (i >= 0);
	}
    };
};


template <typename InputVertices, typename InputNormals, typename InputColors>
class render
{
public:
    InputVertices inputVertices;
    InputNormals inputNormals;
    InputColors inputColors;
    typedef typename thrust::counting_iterator<int> CountingIterator;
    int nVertices, width, height, pixelSize;
    thrust::device_vector<float4> transformedVertices;
    thrust::device_vector<char> frame;
    thrust::device_vector<float> P;
    thrust::device_vector<float> M;
    float* M1;
    float* M2;
    float3 cameraPos;
    thrust::device_vector<float> cameraRot;
    DisplayInfo displayInfo;
    KdTree<thrust::device_vector<float4>::iterator>* kdtree;

    render(InputVertices inputVertices, InputNormals inputNormals, InputColors inputColors, int nVertices, int width, int height) : inputVertices(inputVertices),
	   inputNormals(inputNormals), inputColors(inputColors), nVertices(nVertices), width(width), height(height), pixelSize(4)
    {
      frame.resize(width*height*pixelSize);

      P.resize(16);  M.resize(16);  cameraRot.resize(16);  M1 = new float[16];  M2 = new float[16];
      for (unsigned int i=0; i<4; i++) for (unsigned int j=0; j<4; j++) { P[i*4+j] = (i == j) ? 1.0f : 0.0f;  M[i*4+j] = (i == j) ? 1.0f : 0.0f; }

      kdtree = new KdTree<thrust::device_vector<float4>::iterator>();
    };

    void update(InputVertices ainputVertices, InputNormals ainputNormals, InputColors ainputColors, int anVertices)
    {
	inputVertices = ainputVertices; inputNormals = ainputNormals; inputColors = ainputColors;
	nVertices = anVertices;
    }

    void setOrtho(float left, float right, float bottom, float top, float near, float far)
    {
	for (unsigned int i=0; i<4; i++) for (unsigned int j=0; j<4; j++) P[i*4+j] = 0.0f;
	P[0*4+0] = 2.0f/(right-left);  P[1*4+1] = 2.0f/(top-bottom);  P[2*4+2] = -2.0f/(far-near);  P[3*4+3] = 1.0f;
	P[0*4+3] = -(right+left)/(right-left);
	P[1*4+3] = -(top+bottom)/(top-bottom);
	P[2*4+3] = -(far+near)/(far-near);

	displayInfo.left = left;  displayInfo.right = right;  displayInfo.bottom = bottom;  displayInfo.top = top;  displayInfo.zNear = near;  displayInfo.zFar = far;
	displayInfo.cameraFOV = -1.0f;
    }

    void setPerspective(float fov, float aspect, float near, float far)
    {
	for (unsigned int i=0; i<4; i++) for (unsigned int j=0; j<4; j++) P[i*4+j] = 0.0f;
	float f = 1.0f/tan(0.5*fov*3.14159/180.0);
	P[0*4+0] = f/aspect;
	P[1*4+1] = f;
	P[2*4+2] = (far+near)/(near-far);
	P[2*4+3] = (2*far*near)/(near-far);
	P[3*4+2] = -1.0f;

	displayInfo.cameraFOV = fov;  displayInfo.zNear = near;  displayInfo.zFar = far;
    }

    void setLookAt(float3 eye, float3 center, float3 up)
    {
#ifdef SCANLINE
      float3 F = center - eye;
      float3 f = normalize(F);
      float3 upn = normalize(up);
      float3 s = cross(f, upn);
      float3 u = cross(s, f);

      for (unsigned int i=0; i<4; i++) for (unsigned int j=0; j<4; j++) M1[i*4+j] = 0.0f;
      M1[0*4+0] = s.x;  M1[0*4+1] = s.y;  M1[0*4+2] = s.z;
      M1[1*4+0] = u.x;  M1[1*4+1] = u.y;  M1[1*4+2] = u.z;
      M1[2*4+0] = -f.x;  M1[2*4+1] = -f.y;  M1[2*4+2] = -f.z;
      M1[3*4+3] = 1.0f;

      for (unsigned int i=0; i<4; i++) for (unsigned int j=0; j<4; j++) M2[i*4+j] = 0.0f;
      for (unsigned int i=0; i<4; i++) M2[i*4+i] = 1.0f;
      M2[0*4+3] = -eye.x;  M2[1*4+3] = -eye.y;  M2[2*4+3] = -eye.z;

      float* m = matrixMul(M1, M2);
      M.assign(m, m+16);
#else
      cameraPos.x = eye.x;  cameraPos.y = eye.y;  cameraPos.z = eye.z;
      eye = eye - center;
      eye = normalize(eye);
      up = normalize(up);
      float3 Cy = cross(up, eye);
      Cy = normalize(Cy);
      up = cross(eye, Cy);

      cameraRot[0] = eye.x;  cameraRot[1] = Cy.x;  cameraRot[2] = up.x;  cameraRot[3] = 0.0f;
      cameraRot[4] = eye.y;  cameraRot[5] = Cy.y;  cameraRot[6] = up.y;  cameraRot[7] = 0.0f;
      cameraRot[8] = eye.z;  cameraRot[9] = Cy.z;  cameraRot[10] = up.z; cameraRot[11] = 0.0f;
      cameraRot[12] = 0.0f;  cameraRot[13] = 0.0f; cameraRot[14] = 0.0f; cameraRot[15] = 1.0f;
#endif
    }

    void translate(float x, float y, float z)
    {
      for (unsigned int i=0; i<4; i++) for (unsigned int j=0; j<4; j++) M1[i*4+j] = 0.0f;
      for (unsigned int i=0; i<4; i++) M1[i*4+i] = 1.0f;
      M1[0*4+3] = x;  M1[1*4+3] = y;  M1[2*4+3] = z;

      for (unsigned int i=0; i<16; i++) M2[i] = M[i];
      float* m = matrixMul(M2, M1);
      M.assign(m, m+16);
    }

    void rotate(float* r)
    {
      for (unsigned int i=0; i<16; i++) M2[i] = M[i];
      for (unsigned int i=0; i<16; i++) M1[i] = r[(i%4)*4+(i/4)];
      float* m = matrixMul(M2, M1);
      M.assign(m, m+16);
    }

    void setRot(float* r)
    {
      for (unsigned int i=0; i<16; i++) M[i] = r[(i%4)*4+(i/4)];
    }

    void setLightProperties(float3 alightAmb, float3 alightDif, float akc, float akl, float akq, float4 alp)
    {
      displayInfo.lightAmb = make_float3(alightAmb.x, alightAmb.y, alightAmb.z);
      displayInfo.lightDif = make_float3(alightDif.x, alightDif.y, alightDif.z);
      displayInfo.kc = akc;
      displayInfo.kl = akl;
      displayInfo.kq = akq;
      displayInfo.lp = alp;
    }

    void operator()()
    {
      thrust::fill(frame.begin(), frame.end(), 255);
      thrust::transform(inputVertices, inputVertices+nVertices, inputVertices, vertexTransformModelview(thrust::raw_pointer_cast(&*M.begin())));
      thrust::transform(inputNormals,  inputNormals+nVertices,  inputNormals,  normalTransformModelview(thrust::raw_pointer_cast(&*M.begin())));

#ifndef SCANLINE
#ifdef USE_KD_TREE

#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA
      int levelOffset = 10;
#endif
#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP
      int levelOffset = 7;
#endif

      kdtree->buildTree(inputVertices, nVertices/3, std::max(1.0f, (float)(log(nVertices/3.0f)/log(2.0f)-levelOffset)));
#endif
#endif


#ifdef SCANLINE
      transformedVertices.resize(nVertices);
      thrust::transform(inputVertices, inputVertices+nVertices, transformedVertices.begin(), vertexTransformProjection(width, height, thrust::raw_pointer_cast(&*P.begin())));
      thrust::for_each(CountingIterator(0), CountingIterator(0)+nVertices/3, scanline(inputVertices, transformedVertices.begin(), inputNormals, inputColors, nVertices, width, height,
                                                                                      pixelSize, displayInfo, thrust::raw_pointer_cast(&*frame.begin())));
#else
      thrust::for_each(CountingIterator(0), CountingIterator(0)+width*height, raycast(inputVertices, inputNormals, inputColors, nVertices, width,
                                         height, pixelSize, displayInfo, cameraPos, thrust::raw_pointer_cast(&*cameraRot.begin()),
                                         thrust::raw_pointer_cast(&*(kdtree->tree.begin())), thrust::raw_pointer_cast(&*(kdtree->S.begin())), kdtree->levels,
                                         thrust::raw_pointer_cast(&*frame.begin())));
#endif
    }

    struct scanline : public thrust::unary_function<int, void>
    {
      InputVertices inputVertices;
      thrust::device_vector<float4>::iterator transformedVertices;
      InputNormals inputNormals;
      InputColors inputColors;
      char* frame;
      DisplayInfo displayInfo;
      int nVertices, width, height, pixelSize;

      __host__ __device__
      scanline(InputVertices inputVertices, thrust::device_vector<float4>::iterator transformedVertices, InputNormals inputNormals, InputColors inputColors, int nVertices,
               int width, int height, int pixelSize, DisplayInfo displayInfo, char* frame) : inputVertices(inputVertices), transformedVertices(transformedVertices),
               inputNormals(inputNormals),  inputColors(inputColors), nVertices(nVertices), width(width), height(height), pixelSize(pixelSize), displayInfo(displayInfo), frame(frame) {};

      __host__ __device__
      void operator() (int id) const
      {
        int minY = height;  int maxY = 0;

        float3 n[3];
        for (unsigned int i=0; i<3; i++) n[i] = *(inputNormals + 3*id + i);
        float3 na; na.x = na.y = na.z = 0.0f;
        for (unsigned int i=0; i<3; i++) { na.x += n[i].x;  na.y += n[i].y;  na.z += n[i].z; }
        na = normalize(na);

        float4 L4 = displayInfo.lp - *(inputVertices + 3*id + 0);
        float3 L = make_float3(L4);
        L = normalize(L);

        float dp = -dot(L, na);
        dp = fabs(dp); //if (dp < 0.0f) dp = 0.0f;
        float d = 10.0f;

        float4 curColor = *(inputColors + 3*id + 0);
        curColor.x = displayInfo.lightAmb.x*curColor.x + (1.0/(displayInfo.kc+displayInfo.kl*d+displayInfo.kq*d*d))*1*(displayInfo.lightAmb.x*curColor.x + dp*displayInfo.lightDif.x*curColor.x);
        curColor.y = displayInfo.lightAmb.y*curColor.y + (1.0/(displayInfo.kc+displayInfo.kl*d+displayInfo.kq*d*d))*1*(displayInfo.lightAmb.y*curColor.y + dp*displayInfo.lightDif.y*curColor.y);
        curColor.z = displayInfo.lightAmb.z*curColor.z + (1.0/(displayInfo.kc+displayInfo.kl*d+displayInfo.kq*d*d))*1*(displayInfo.lightAmb.z*curColor.z + dp*displayInfo.lightDif.z*curColor.z);

        //std::cout << dp << " " << L.x << " " << L.y << " " << L.z << " " << na.x << " " << na.y << " " << na.z << std::endl;
        //std::cout << curColor.x << " " << curColor.y << " " << curColor.z << std::endl;

        if (curColor.x > 1.0f) curColor.x = 1.0f;
        if (curColor.y > 1.0f) curColor.y = 1.0f;
        if (curColor.z > 1.0f) curColor.z = 1.0f;

        int b = curColor.z*255;
        int g = curColor.y*255;
        int r = curColor.x*255;

        for (unsigned int v=0; v<3; v++)
        {
          float3 vertex = make_float3(*(transformedVertices + 3*id + v));
          int i = (int)(vertex.y);   int j = (int)(vertex.x);
          frame[i*height*pixelSize + j*pixelSize + 0] = b;
          frame[i*height*pixelSize + j*pixelSize + 1] = g;
          frame[i*height*pixelSize + j*pixelSize + 2] = r;
          if (i < minY) minY = i;  if (i > maxY) maxY = i;
        }

        //std::cout << "Range " << minY << " " << maxY << std::endl;
        for (unsigned int s=minY; s<=maxY; s++)
        {
          float3 v0 = make_float3(*(transformedVertices + 3*id + 0));
          float3 v1 = make_float3(*(transformedVertices + 3*id + 1));
          float3 v2 = make_float3(*(transformedVertices + 3*id + 2));

          int x0 = -99999;
          if ((((s > v0.y) && (s < v1.y)) || ((s < v0.y) && (s > v1.y))) && (fabs(v0.x-v1.x) > 0.00001))
          {
            float m0 = (v0.y - v1.y) / (v0.x - v1.x);
            float b0 = v0.y - m0*v0.x;
            x0 = (int)((s - b0)/m0);
          }

          int x1 = -99999;
          if ((((s > v2.y) && (s < v1.y)) || ((s < v2.y) && (s > v1.y))) && (fabs(v2.x-v1.x) > 0.00001))
          {
            float m1 = (v2.y - v1.y) / (v2.x - v1.x);
            float b1 = v2.y - m1*v2.x;
            x1 = (int)((s - b1)/m1);
          }

          int x2 = -99999;
          if ((((s > v0.y) && (s < v2.y)) || ((s < v0.y) && (s > v2.y))) && (fabs(v0.x-v2.x) > 0.00001))
          {
            float m2 = (v0.y - v2.y) / (v0.x - v2.x);
            float b2 = v0.y - m2*v0.x;
            x2 = (int)((s - b2)/m2);
          }

          int sb = 99999;
          if ((x0 > -9999) && (x0 < sb)) sb = x0;
          if ((x1 > -9999) && (x1 < sb)) sb = x1;
          if ((x2 > -9999) && (x2 < sb)) sb = x2;

          int ss = -99999;
          if ((x0 > -9999) && (x0 > ss)) ss = x0;
          if ((x1 > -9999) && (x1 > ss)) ss = x1;
          if ((x2 > -9999) && (x2 > ss)) ss = x2;

          if ((sb >= 0) && (sb < width) && (ss >= 0) && (ss < width))
          {
            for (unsigned int f=sb; f<=ss; f++)
            {
              frame[s*height*pixelSize + f*pixelSize + 0] = b;
              frame[s*height*pixelSize + f*pixelSize + 1] = g;
              frame[s*height*pixelSize + f*pixelSize + 2] = r;
            }
          }
        }
      }
    };

    struct vertexTransformModelview : public thrust::unary_function<float4, float4>
    {
    	float* M;

    	__host__ __device__
    	vertexTransformModelview(float* M) : M(M) {};

    	__host__ __device__
    	float4 operator() (float4 v)
    	{
    	  float4 vw = make_float4(v.x, v.y, v.z, v.w);
    	  vw = matrixMul(M, vw);
    	  return vw;
    	}
    };

    struct normalTransformModelview : public thrust::unary_function<float3, float3>
    {
        float* M;

        __host__ __device__
        normalTransformModelview(float* M) : M(M) {};

        __host__ __device__
        float3 operator() (float3 v)
        {
          float3 vw = matrixMul(M, v);
          return vw;
        }
    };

    struct vertexTransformProjection : public thrust::unary_function<float4, float4>
    {
	float* P;
	int width, height;

	__host__ __device__
	vertexTransformProjection(int width, int height, float* P) : width(width), height(height), P(P) {};

	__host__ __device__
	float4 operator() (float4 v)
	{
	  float4 vw = make_float4(v.x, v.y, v.z, v.w);
	  vw = matrixMul(P, vw);
	  v = make_float4(vw.x/vw.w, vw.y/vw.w, vw.z/vw.w, 1.0f);
	  v.x = width *(v.x + 1)*0.5;
	  v.y = height*(v.y + 1)*0.5;
	  v.z = width *(v.z + 1)*0.5;
	  return v;
	}
    };

    struct raycast : public thrust::unary_function<int, void>
    {
      InputVertices inputVertices;
      InputNormals inputNormals;
      InputColors inputColors;
      char* frame;
      int nVertices, width, height, pixelSize;
      float3 cameraPos;
      float* cameraRot;
      DisplayInfo displayInfo;
      TreeNode* tree;
      int* treeTris;
      int numLevels;

      __host__ __device__
      raycast(InputVertices inputVertices, InputNormals inputNormals, InputColors inputColors, int nVertices,
              int width, int height, int pixelSize, DisplayInfo displayInfo, float3 cameraPos, float* cameraRot, TreeNode* tree, int* treeTris, int numLevels, char* frame) :
              inputVertices(inputVertices), inputNormals(inputNormals), inputColors(inputColors), nVertices(nVertices), width(width),
              height(height), pixelSize(pixelSize), displayInfo(displayInfo), cameraPos(cameraPos), cameraRot(cameraRot), tree(tree), treeTris(treeTris), numLevels(numLevels),
              frame(frame) {};

      __host__ __device__
      bool intersectionSegmentTriangle(float3& a_segmentPointA, float3& a_segmentPointB, float3& a_triangleVertex0, float3& a_triangleVertex1, float3& a_triangleVertex2,
                                       float3& a_collisionPoint, float3& a_collisionNormal)
      {
        // This value controls how close rays can be to parallel to the triangle
	// surface before we discard them
	const double CHAI_INTERSECT_EPSILON = 10e-14f;

	// compute a ray and check its length
	float3 rayDir;
	rayDir = a_segmentPointB - a_segmentPointA;
	double segmentLengthSquare = dot(rayDir, rayDir);
	if (segmentLengthSquare == 0.0) { return (false); }

	// Compute the triangle's normal
	float3 t_E0, t_E1, t_N;

	t_E0 = a_triangleVertex1 - a_triangleVertex0;
	t_E1 = a_triangleVertex2 - a_triangleVertex0;
	t_N = cross(t_E0, t_E1);

	// If the ray is parallel to the triangle (perpendicular to the
        // normal), there's no collision
	if (fabs(dot(t_N, rayDir))<10E-15f) return (false);

	double t_T = dot(t_N, a_triangleVertex0 - a_segmentPointA) / dot(t_N, rayDir);

	if (t_T + CHAI_INTERSECT_EPSILON < 0) return (false);

	float3 t_Q = a_segmentPointA + (t_T * rayDir) - a_triangleVertex0;
	double t_Q0 = dot(t_E0,t_Q);
	double t_Q1 = dot(t_E1,t_Q);
	double t_E00 = dot(t_E0,t_E0);
	double t_E01 = dot(t_E0,t_E1);
	double t_E11 = dot(t_E1,t_E1);
	double t_D = (t_E00 * t_E11) - (t_E01 * t_E01);

	if ((t_D > -CHAI_INTERSECT_EPSILON) && (t_D < CHAI_INTERSECT_EPSILON)) return(false);

	double t_S0 = ((t_E11 * t_Q0) - (t_E01 * t_Q1)) / t_D;
	double t_S1 = ((t_E00 * t_Q1) - (t_E01 * t_Q0)) / t_D;

	// Collision has occurred. It is reported.
	if ((t_S0 >= 0.0 - CHAI_INTERSECT_EPSILON) && (t_S1 >= 0.0 - CHAI_INTERSECT_EPSILON) &&
	    ((t_S0 + t_S1) <= 1.0 + CHAI_INTERSECT_EPSILON))
	{
	  float3 t_I = a_triangleVertex0 + (t_S0 * t_E0) + (t_S1 * t_E1);

	  // Square distance between ray origin and collision point.
	  double distanceSquare = dot(a_segmentPointA - t_I, a_segmentPointA - t_I);

	  // check if collision occurred within segment. If yes, report collision
	  if (distanceSquare <= segmentLengthSquare)
	  {
	    a_collisionPoint = a_segmentPointA + (t_T * rayDir);
	    a_collisionNormal = normalize(t_N);
	    if (cosAngle(a_collisionNormal, rayDir) > 0.0) a_collisionNormal = -1.0f * a_collisionNormal;
	    return (true);
	  }
	}

	// no collision occurred
	return (false);
      }

      __host__ __device__
      bool searchTriangle(int t, float3 r0, float3 r1, int x, int y, float& maxZ)
      {
	float3 v[3];
	for (unsigned int i=0; i<3; i++) v[i] = make_float3(*(inputVertices + 3*t + i));

	float3 ipt, in;
	bool isect = intersectionSegmentTriangle(r0, r1, v[0], v[1], v[2], ipt, in);

	if ((isect) && (ipt.z > maxZ) /*&& (ipt.z > 128)*/)
	{
	  float3 n[3];
	  for (unsigned int i=0; i<3; i++) n[i] = *(inputNormals + 3*t + i);
	  float3 na; na.x = na.y = na.z = 0.0f;
	  for (unsigned int i=0; i<3; i++) { na.x += n[i].x;  na.y += n[i].y;  na.z += n[i].z; }
	  na = normalize(na);

	  float4 L4 = displayInfo.lp - *(inputVertices + 3*t + 0);
	  float3 L = make_float3(L4);
	  L = normalize(L);

	  float dp = -dot(L, na);
	  dp = fabs(dp); //if (dp < 0.0f) dp = 0.0f;
	  float d = 10.0f;

	  float4 curColor = *(inputColors + 3*t + 0);
	  curColor.x = displayInfo.lightAmb.x*curColor.x + (1.0/(displayInfo.kc+displayInfo.kl*d+displayInfo.kq*d*d))*1*(displayInfo.lightAmb.x*curColor.x + dp*displayInfo.lightDif.x*curColor.x);
	  curColor.y = displayInfo.lightAmb.y*curColor.y + (1.0/(displayInfo.kc+displayInfo.kl*d+displayInfo.kq*d*d))*1*(displayInfo.lightAmb.y*curColor.y + dp*displayInfo.lightDif.y*curColor.y);
	  curColor.z = displayInfo.lightAmb.z*curColor.z + (1.0/(displayInfo.kc+displayInfo.kl*d+displayInfo.kq*d*d))*1*(displayInfo.lightAmb.z*curColor.z + dp*displayInfo.lightDif.z*curColor.z);

	  if (curColor.x > 1.0f) curColor.x = 1.0f;
	  if (curColor.y > 1.0f) curColor.y = 1.0f;
	  if (curColor.z > 1.0f) curColor.z = 1.0f;

	  int b = curColor.z*255;
	  int g = curColor.y*255;
	  int r = curColor.x*255;

          frame[y*height*pixelSize + x*pixelSize + 0] = b;
	  frame[y*height*pixelSize + x*pixelSize + 1] = g;
	  frame[y*height*pixelSize + x*pixelSize + 2] = r;
	  maxZ = ipt.z;
        }
        return isect;
      }

      __host__ __device__
      void operator() (int id)
      {
        int x = id / width;  int y = id % width;

        float3 r0, r1;
        if (displayInfo.cameraFOV > 0.0f)
        {
          double distCam = (height / 2.0f) / tan(0.5*3.1415*displayInfo.cameraFOV / 180.0f);
          float3 selectRay;
          selectRay.x = -distCam;  selectRay.y = (x - (width / 2.0f));  selectRay.z = y - (height / 2.0f);
          selectRay = normalize(selectRay);
          selectRay = matrixMul(cameraRot, selectRay);

          r0 = make_float3(cameraPos.x, cameraPos.y, cameraPos.z);
          r1 = make_float3(r0.x + 100000*selectRay.x, r0.y + 100000*selectRay.y, r0.z + 100000*selectRay.z);
        }
        else
        {
          float3 selectRay;
          selectRay.x = displayInfo.left + (1.0f*x/width)*(displayInfo.right-displayInfo.left);  selectRay.y = displayInfo.bottom + (1.0f*y/height)*(displayInfo.top-displayInfo.bottom);
          r0 = make_float3(selectRay.x, selectRay.y, displayInfo.zNear);
          r1 = make_float3(selectRay.x, selectRay.y, displayInfo.zFar);
        }
        float maxZ = -FLT_MAX;

#ifdef USE_KD_TREE
        int maxLevel = numLevels;
        const int numNodes = 2;
        int processNodes[numNodes];
        processNodes[0] = 1;  for (unsigned int i=1; i<numNodes; i++) processNodes[i] = -1;
        bool nodesToProcess = true;
        int tail = 1;
        while (nodesToProcess)
        {
          int curIndex = processNodes[0];
          for (unsigned int i=0; i<numNodes-1; i++) processNodes[i] = processNodes[i+1];  processNodes[numNodes-1] = -1;
          for (unsigned int i=numNodes-1; i>=1; i--) if (processNodes[i] == -1) tail = i;  if (processNodes[0] == -1) tail = 0;
          //std::cout << tail;
          if (tree[curIndex-1].level < maxLevel-1)
          {
            if (tree[curIndex-1].splitDim == 0) { if (r0.x < tree[curIndex-1].splitVal) processNodes[tail] = 2*curIndex; else processNodes[tail] = 2*curIndex+1; }
            if (tree[curIndex-1].splitDim == 1) { if (r0.y < tree[curIndex-1].splitVal) processNodes[tail] = 2*curIndex; else processNodes[tail] = 2*curIndex+1; }
          }
          else
          {
            for (unsigned int i=0; i<tree[curIndex-1].numTris; i++)
              searchTriangle(treeTris[tree[curIndex-1].startIndex+i], r0, r1, x, y, maxZ);
          }
          nodesToProcess = false;  for (unsigned int i=0; i<numNodes; i++) if (processNodes[i] >= 0) nodesToProcess = true;
        }
#else
        for (int t = 0; t < nVertices/3; t++) searchTriangle(t, r0, r1, x, y, maxZ);
#endif
      }
    };

    thrust::device_vector<char>::iterator frame_begin() { return frame.begin(); }
    thrust::device_vector<char>::iterator frame_end() { return frame.end(); }
};

}

#endif /* RENDER_H_ */
