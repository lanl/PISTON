/*
Copyright (c) 2011, Los Alamos National Security, LLC
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
    	and/or other materials provided with the distribution.
    Neither the name of the Los Alamos National Laboratory nor the names of its contributors may be used to endorse or promote products derived from this
    	software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
#include <cutil_math.h>
#include <piston/choose_container.h>


namespace piston {

template <typename InputPoints, typename InputVectors, typename GlyphVertices, typename GlyphNormals, typename GlyphIndices>
class glyph
{
public:
	InputPoints inputPoints;
	InputVectors inputVectors;
	GlyphVertices glyphVertices;
	GlyphNormals glyphNormals;
	GlyphIndices glyphIndices;

	int nPoints, nVertices, nIndices;

	typedef typename detail::choose_container<GlyphVertices, float>::type 	VerticesContainer;
	typedef typename detail::choose_container<GlyphNormals, float>::type	NormalsContainer;
	typedef typename detail::choose_container<GlyphIndices, int>::type	IndicesContainer;

	typedef typename VerticesContainer::iterator VerticesIterator;
	typedef typename IndicesContainer::iterator  IndicesIterator;
	typedef typename NormalsContainer::iterator  NormalsIterator;

	typedef typename thrust::counting_iterator<int>	CountingIterator;

	VerticesContainer	vertices;
	NormalsContainer	normals;
	IndicesContainer	indices;


	glyph(InputPoints inputPoints, InputVectors inputVectors, GlyphVertices glyphVertices, GlyphNormals glyphNormals, GlyphIndices glyphIndices,
		  int nPoints, int nVertices, int nIndices) : inputPoints(inputPoints), inputVectors(inputVectors), glyphVertices(glyphVertices), glyphNormals(glyphNormals),
		  glyphIndices(glyphIndices), nPoints(nPoints), nVertices(nVertices), nIndices(nIndices) {};

	void operator()()
	{
	  int NCells = 1;
	  vertices.resize(nVertices);
	  normals.resize(nVertices);
	  indices.resize(nIndices);
	  thrust::for_each(CountingIterator(0), CountingIterator(0)+NCells, generate_glyphs(inputPoints, inputVectors, glyphVertices, glyphNormals, glyphIndices,
			  thrust::raw_pointer_cast(&*vertices.begin()), thrust::raw_pointer_cast(&*normals.begin()), thrust::raw_pointer_cast(&*indices.begin()), nVertices, nIndices));
	}

	struct generate_glyphs : public thrust::unary_function<int, void>
	{
	  InputPoints inputPoints;
	  InputVectors inputVectors;
	  GlyphVertices glyphVertices;
	  GlyphNormals glyphNormals;
	  GlyphIndices glyphIndices;

	  float* vertices;
	  float* normals;
	  int* indices;

	  int nVertices, nIndices;

	  __host__ __device__
	  generate_glyphs(InputPoints inputPoints, InputVectors inputVectors, GlyphVertices glyphVertices, GlyphNormals glyphNormals, GlyphIndices glyphIndices,
			          float* vertices, float* normals, int* indices, int nVertices, int nIndices) : inputPoints(inputPoints), inputVectors(inputVectors), glyphVertices(glyphVertices),
			          glyphNormals(glyphNormals), glyphIndices(glyphIndices), vertices(vertices), normals(normals), indices(indices), nVertices(nVertices), nIndices(nIndices) {};

	  __host__ __device__
	  void operator() (int id) const
	  {
        for (int i=0; i<nVertices; i++) *(vertices+i) = *(glyphVertices+i);
        for (int i=0; i<nVertices; i++)  normals[i] = glyphNormals[i];
        for (int i=0; i<nIndices; i++)  *(indices+i) = *(glyphIndices+i);
	  }
	};

	VerticesIterator vertices_begin()  { return vertices.begin(); }
	VerticesIterator vertices_end()    { return vertices.end();   }
	NormalsIterator normals_begin()    { return normals.begin();  }
	NormalsIterator normals_end()      { return normals.end();    }
	IndicesIterator indices_begin()    { return indices.begin();  }
	IndicesIterator indices_end()      { return indices.end();    }
};

}

#endif /* GLYPH_H_ */
