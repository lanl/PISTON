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

#ifndef HSV_COLOR_MAP_H_
#define HSV_COLOR_MAP_H_

namespace piston
{

template <typename ValueType>
struct color_map : thrust::unary_function<ValueType, float4>
{
    const ValueType min;
    const ValueType max;
    const ValueType blackMin;
    const ValueType whiteMax;
    const bool reversed;

    __host__ __device__
    color_map(ValueType min, ValueType max, bool reversed=false, ValueType blackMin=-1000000.0, ValueType whiteMax=1000000.0) :
	min(min), max(max), reversed(reversed), blackMin(blackMin), whiteMax(whiteMax) {}

    __host__ __device__
    float4 operator()(ValueType val) {
	// HSV rainbow for height field, stolen form Manta
    if (val < blackMin) return make_float4(0.0, 0.0, 0.0, 1.0);
    if (val > whiteMax) return make_float4(1.0, 1.0, 1.0, 1.0);
	const float V = 0.7f, S = 1.0f;
	float H = (1.0f - static_cast<float> (val - min) / (max - min));
	if (reversed) H = 1.0 - H;

	if (H < 0.0f)
	    H = 0.0f;
	else if (H > 1.0f)
	    H = 1.0f;
	H *= 4.0f;

	float i = floor(H);
	float f = H - i;

	float p = V * (1.0 - S);
	float q = V * (1.0 - S * f);
	float t = V * (1.0 - S * (1 - f));

	float R, G, B;
	if (i == 0.0) {
	    R = V;
	    G = t;
	    B = p;
	} else if (i == 1.0) {
	    R = q;
	    G = V;
	    B = p;
	} else if (i == 2.0) {
	    R = p;
	    G = V;
	    B = t;
	} else if (i == 3.0) {
	    R = p;
	    G = q;
	    B = V;
	} else if (i == 4.0) {
	    R = t;
	    G = p;
	    B = V;
	} else {
	    // i == 5.0
	    R = V;
	    G = p;
	    B = q;
	}
	return make_float4(R, G, B, 1.0);
    }
};

}

#endif /* HSV_COLOR_MAP_H_ */
