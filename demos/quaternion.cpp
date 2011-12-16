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

This file (quaternion.cpp) contains code derived from this example on the web: http://content.gpwiki.org/index.php/OpenGL:Tutorials:Using_Quaternions_to_represent_rotation,
which is re-distributed under this licence: http://www.gnu.org/licenses/old-licenses/fdl-1.2.txt .
*/

#include "quaternion.h"

#define TOLERANCE 0.00001f
#define PIOVER180 3.14159/180.0

void Quaternion::normalise()
{
	// Don't normalize if we don't have to
	float mag2 = w * w + x * x + y * y + z * z;
	if (fabs(mag2) > TOLERANCE && fabs(mag2 - 1.0f) > TOLERANCE) {
		float mag = sqrt(mag2);
		w /= mag;
		x /= mag;
		y /= mag;
		z /= mag;
	}
}


Quaternion Quaternion::getConjugate()
{
	return Quaternion(-x, -y, -z, w);
}


Quaternion Quaternion::operator* (const Quaternion &rq)
{
	// the constructor takes its arguments as (x, y, z, w)
	return Quaternion(w * rq.x + x * rq.w + y * rq.z - z * rq.y,
	                  w * rq.y + y * rq.w + z * rq.x - x * rq.z,
	                  w * rq.z + z * rq.w + x * rq.y - y * rq.x,
	                  w * rq.w - x * rq.x - y * rq.y - z * rq.z);
}


float3 Quaternion::operator* (const float3 &vec)
{
	float3 vn(vec);
	float mag2 = vn.x * vn.x + vn.y * vn.y + vn.z * vn.z;
	float mag = sqrt(mag2);
	if (fabs(mag) > TOLERANCE)
	{
	  vn.x /= mag;
	  vn.y /= mag;
	  vn.z /= mag;
	}
	//vn.normalise();

	Quaternion vecQuat, resQuat;
	vecQuat.x = vn.x;
	vecQuat.y = vn.y;
	vecQuat.z = vn.z;
	vecQuat.w = 0.0f;

	resQuat = vecQuat * getConjugate();
	resQuat = *this * resQuat;

	float3 result;  result.x = resQuat.x;  result.y = resQuat.y;  result.z = resQuat.z;
	return (result);
}


void Quaternion::FromAxis(const float3 &v, float angle)
{
	float sinAngle;
	angle *= 0.5f;
	float3 vn(v);
	float mag2 = vn.x * vn.x + vn.y * vn.y + vn.z * vn.z;
	float mag = sqrt(mag2);
	if (fabs(mag) > TOLERANCE)
	{
	  vn.x /= mag;
	  vn.y /= mag;
	  vn.z /= mag;
	}
	//vn.normalise();

	sinAngle = sin(angle);

	x = (vn.x * sinAngle);
	y = (vn.y * sinAngle);
	z = (vn.z * sinAngle);
	w = cos(angle);
}


void Quaternion::FromEuler(float pitch, float yaw, float roll)
{
	// Basically we create 3 Quaternions, one for pitch, one for yaw, one for roll
	// and multiply those together.
	// the calculation below does the same, just shorter

	float p = pitch * PIOVER180 / 2.0;
	float y = yaw * PIOVER180 / 2.0;
	float r = roll * PIOVER180 / 2.0;

	float sinp = sin(p);
	float siny = sin(y);
	float sinr = sin(r);
	float cosp = cos(p);
	float cosy = cos(y);
	float cosr = cos(r);

	this->x = sinr * cosp * cosy - cosr * sinp * siny;
	this->y = cosr * sinp * cosy + sinr * cosp * siny;
	this->z = cosr * cosp * siny - sinr * sinp * cosy;
	this->w = cosr * cosp * cosy + sinr * sinp * siny;

	normalise();
}


void Quaternion::getMatrix(float* m) const
{
	float x2 = x * x;
	float y2 = y * y;
	float z2 = z * z;
	float xy = x * y;
	float xz = x * z;
	float yz = y * z;
	float wx = w * x;
	float wy = w * y;
	float wz = w * z;

	// This calculation would be a lot more complicated for non-unit length quaternions
	// Note: The constructor of Matrix4 expects the Matrix in column-major format like expected by
	//   OpenGL
	m[0] = 1.0f - 2.0f * (y2 + z2);  m[1] = 2.0f * (xy - wz);  m[2] = 2.0f * (xz + wy);  m[3] = 0.0f;
	m[4] = 2.0f * (xy + wz);  m[5] = 1.0f - 2.0f * (x2 + z2);  m[6] = 2.0f * (yz - wx);  m[7] = 0.0f;
	m[8] = 2.0f * (xz - wy);  m[9] = 2.0f * (yz + wx);  m[10] = 1.0f - 2.0f * (x2 + y2);  m[11] = 0.0f;
	m[12] = 0.0f;  m[13] = 0.0f;  m[14] = 0.0f;  m[15] = 1.0f;
}


void Quaternion::getAxisAngle(float3 *axis, float *angle)
{
	float scale = sqrt(x * x + y * y + z * z);
	axis->x = x / scale;
	axis->y = y / scale;
	axis->z = z / scale;
	*angle = acos(w) * 2.0f;
}







