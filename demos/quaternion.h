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

This file (quaternion.h) contains code derived from this example on the web: http://content.gpwiki.org/index.php/OpenGL:Tutorials:Using_Quaternions_to_represent_rotation,
which is re-distributed under this licence: http://www.gnu.org/licenses/old-licenses/fdl-1.2.txt .
*/

#include <math.h>
#include <stdlib.h>

#include <thrust/host_vector.h>

class Quaternion
{
public:
	Quaternion() { x = y = z = w = 0.0; }
	Quaternion(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) {};
	void set(double ax, double ay, double az, double aw) { x = ax; y = ay; z = az; w = aw; }
	void normalise();
	Quaternion getConjugate();
	Quaternion operator* (const Quaternion &rq);
	float3 operator* (const float3 &vec);
	void FromAxis(const float3 &v, float angle);
	void FromEuler(float pitch, float yaw, float roll);
	void getMatrix(float* m) const;
	void getAxisAngle(float3 *axis, float *angle);

	double x,y,z,w;
};
