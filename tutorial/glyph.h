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

#ifndef GLYPH_H_
#define GLYPH_H_

#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <piston/image3d.h>
#include <piston/piston_math.h>
#include <piston/choose_container.h>


//===========================================================================
/*!    
    \class      glyph

    \brief      
    Computes a glyph at each input point, oriented, scaled and colors according
    to input vectors and scalars
*/
//===========================================================================
template <typename InputPoints, typename InputVectors, typename InputScalars, typename GlyphVertices, typename GlyphNormals, typename GlyphIndices>
class glyph
{
  public:

    //==========================================================================
    /*! 
        struct generate_glyphs

        Compute the glyph at the given input point
    */
    //==========================================================================
    struct generate_glyphs : public thrust::unary_function<int, void>
    {
 	InputPoints inputPoints;
	InputVectors inputVectors;
	InputScalars inputScalars;
	GlyphVertices glyphVertices;
	GlyphNormals glyphNormals;
	GlyphIndices glyphIndices;

	float3 *vertices, *normals;
	uint3* indices;
	float* scalars;
	int nVertices, nIndices;

	__host__ __device__
	generate_glyphs(InputPoints inputPoints, InputVectors inputVectors, InputScalars inputScalars, GlyphVertices glyphVertices, GlyphNormals glyphNormals, GlyphIndices glyphIndices,
			float3* vertices, float3* normals, uint3* indices, float* scalars, int nVertices, int nIndices) : 
                        inputPoints(inputPoints), inputVectors(inputVectors), inputScalars(inputScalars),
			glyphVertices(glyphVertices), glyphNormals(glyphNormals), glyphIndices(glyphIndices), vertices(vertices), normals(normals), indices(indices), scalars(scalars),
			nVertices(nVertices), nIndices(nIndices) {};

	__host__ __device__
	void operator() (int id) const
	{
            // Compute a reference frame with the first column oriented along the given vector, and the other two columns mutually perpendicular
	    float R[9]; 
            float3 col1, col2, col3;
	    col1 = *(inputVectors+id);
            if ((fabs(col1.x) > 0.00001f) || (fabs(col1.y) > 0.00001f)) { col2.x = col1.y;  col2.y = -col1.x; col2.z = 0.0; }
	    else { col2.x = 0.0;  col2.y = 1.0;  col2.z = 0.0; }
            col1 = normalize(col1);  col2 = normalize(col2);
            col3 = cross(col1, col2);  col3 = normalize(col3);
            setMatrix9Columns(R, col1, col2, col3);

            // For the current copy of the glyph, loop through all its vertices
	    float3 base = *(inputPoints+id);
	    for (int i=id*(nVertices); i<(id+1)*(nVertices); i++)
	    {
              // Rotate the glyph vertices according to the reference frame derived from the given vector and offset to the given point
	      float3 glyphV = *(glyphVertices+(i%(nVertices)));
              float3 result = *(inputScalars+id)*matrix9Mul(R, glyphV) + base;
	      *(vertices+i) = result;

              // Rotate the glyph normals according to the current reference frame
	      float3 glyphN = *(glyphNormals+(i%(nVertices)));
              result = matrix9Mul(R, glyphN);
	      *(normals+i) = result;

              // Set the scalar value at each vertex of this copy of the glyph to the given value
	      *(scalars+i) = *(inputScalars+id);
	    }

            // Set the vertex indices for all vertices for this copy of the glyph
            uint3 offset; offset.x = offset.y = offset.z = id*nVertices;
            for (int i=id*nIndices; i<(id+1)*nIndices; i++)  *(indices+i) = (uint3)(*(glyphIndices+(i%nIndices))) + offset;
	}
    };


    //==========================================================================
    /*! 
        Member variable declarations
    */
    //==========================================================================

    //! Output vertices and normals
    thrust::device_vector<float3> vertices, normals;
    //! Output indices
    thrust::device_vector<uint3> indices;
    //! Output scalars
    thrust::device_vector<float> scalars;

    //! Vertex buffers for vertices, normals, colors, and indices, used for interop
    #ifdef USE_INTEROP
      struct cudaGraphicsResource* vboResources[4];      
      float3 *vertexBufferData;
      float3 *normalBufferData;
      float4 *colorBufferData;
      uint3 *indexBufferData;
    #endif


    //==========================================================================
    /*! 
        Constructor for glyph class

        \fn	glyph::glyph
    */
    //==========================================================================
    glyph() { };


    //==========================================================================
    /*! 
        Compute the given glyphs at the given points, orienting along the given 
        vectors and scaling and coloring according to the given scalars

        \fn	flock_sim::operator
    */
    //==========================================================================
    void operator()(InputPoints a_inputPoints, InputVectors a_inputVectors, InputScalars a_inputScalars, GlyphVertices a_glyphVertices, GlyphNormals a_glyphNormals, GlyphIndices a_glyphIndices,
                    int a_numPoints, int a_numVertices, int a_numIndices, float a_colorMapMin=0.0f, float a_colorMapMax=1.0f)
    {
        // Allocate memory as needed for the device vectors
        int outputVertexSize = a_numPoints*a_numVertices; 
        int outputIndexSize = a_numPoints*a_numIndices; 
        vertices.resize(outputVertexSize);
	normals.resize(outputVertexSize);
	scalars.resize(outputVertexSize);
	indices.resize(outputIndexSize);

        // If using interop, call the generate_glyphs functor for_each input point, sending it the vertex buffer objects into which to write its vertex and normals output
        #ifdef USE_INTEROP	  
	  size_t num_bytes;
	  cudaGraphicsMapResources(1, &vboResources[0], 0);
	  cudaGraphicsResourceGetMappedPointer((void **)&vertexBufferData, &num_bytes, vboResources[0]);
	  cudaGraphicsMapResources(1, &vboResources[1], 0);
	  cudaGraphicsResourceGetMappedPointer((void **)&colorBufferData, &num_bytes, vboResources[1]);
	  cudaGraphicsMapResources(1, &vboResources[2], 0);
	  cudaGraphicsResourceGetMappedPointer((void **)&normalBufferData, &num_bytes, vboResources[2]);
	  cudaGraphicsMapResources(1, &vboResources[3], 0);
	  cudaGraphicsResourceGetMappedPointer((void **)&indexBufferData, &num_bytes, vboResources[3]);
	  
    	  thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+a_numPoints, generate_glyphs(
                           a_inputPoints, a_inputVectors, a_inputScalars, 
                           a_glyphVertices, a_glyphNormals, a_glyphIndices,
    	  	           vertexBufferData, normalBufferData, indexBufferData, thrust::raw_pointer_cast(&*scalars.begin()), a_numVertices, a_numIndices));
	    
          if (vboResources[1]) thrust::transform(scalars.begin(), scalars.end(), thrust::device_ptr<float4>(colorBufferData), color_map<float>(a_colorMapMin, a_colorMapMax));
          for (int i=0; i<4; i++) cudaGraphicsUnmapResources(1, &vboResources[i], 0);

        // Otherwise, call the generate_glyphs functor for_each input point, sending it the arrays into which to write its vertex and normals output
        #else
	  thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+a_numPoints, generate_glyphs(
                           a_inputPoints, a_inputVectors, a_inputScalars, 
                           a_glyphVertices, a_glyphNormals, a_glyphIndices,
			   thrust::raw_pointer_cast(&*vertices.begin()), thrust::raw_pointer_cast(&*normals.begin()), thrust::raw_pointer_cast(&*indices.begin()),
			   thrust::raw_pointer_cast(&*scalars.begin()), a_numVertices, a_numIndices));
        #endif
    }


    //==========================================================================
    /*! 
        Accessor functions for vertices, normals, vertex indices, and scalars
    */
    //==========================================================================
    thrust::device_vector<float3>::iterator vertices_begin()  { return vertices.begin(); }
    thrust::device_vector<float3>::iterator vertices_end()    { return vertices.end();   }
    thrust::device_vector<float3>::iterator normals_begin()   { return normals.begin();  }
    thrust::device_vector<float3>::iterator normals_end()     { return normals.end();    }
    thrust::device_vector<uint3>::iterator indices_begin()    { return indices.begin();  }
    thrust::device_vector<uint3>::iterator indices_end()      { return indices.end();    }
    thrust::device_vector<float>::iterator scalars_begin()    { return scalars.begin();  }
    thrust::device_vector<float>::iterator scalars_end()      { return scalars.end();    }
};

#endif /* GLYPH_H_ */
