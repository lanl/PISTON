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

#include "quaternion.h"


void Quaternion::mul(Quaternion q)
{
	float tx, ty, tz, tw;
	tx = w*q.x + x*q.w + y*q.z - z*q.y;
	ty = w*q.y + y*q.w + z*q.x - x*q.z;
	tz = w*q.z + z*q.w + x*q.y - y*q.x;
	tw = w*q.w - x*q.x - y*q.y - z*q.z;

	x = tx; y = ty; z = tz; w = tw;
}


void Quaternion::setEulerAngles(float pitch, float yaw, float roll)
{
	w = cos(pitch/2.0)*cos(yaw/2.0)*cos(roll/2.0) - sin(pitch/2.0)*sin(yaw/2.0)*sin(roll/2.0);
	x = sin(pitch/2.0)*sin(yaw/2.0)*cos(roll/2.0) + cos(pitch/2.0)*cos(yaw/2.0)*sin(roll/2.0);
	y = sin(pitch/2.0)*cos(yaw/2.0)*cos(roll/2.0) + cos(pitch/2.0)*sin(yaw/2.0)*sin(roll/2.0);
	z = cos(pitch/2.0)*sin(yaw/2.0)*cos(roll/2.0) - sin(pitch/2.0)*cos(yaw/2.0)*sin(roll/2.0);

	float norm = sqrt(x*x + y*y + z*z + w*w);
	if (norm > 0.00001) { x /= norm;  y /= norm;  z /= norm;  w /= norm; }
}


void Quaternion::getRotMat(float* m) const
{
	for (int i=0; i<16; i++) m[i] = 0.0; m[15] = 1.0;
	m[0]  = 1 - 2*y*y - 2*z*z;  m[1]  = 2*x*y - 2*z*w;      m[2]  = 2*x*z + 2*y*w;
	m[4]  = 2*x*y + 2*z*w;      m[5]  = 1 - 2*x*x - 2*z*z;  m[6]  = 2*y*z - 2*x*w;
	m[8]  = 2*x*z - 2*y*w;      m[9]  = 2*y*z + 2*x*w;      m[10] = 1 - 2*x*x - 2*y*y;
}








