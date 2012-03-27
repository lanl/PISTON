
#ifndef PISTON_MATH
#define PISTON_MATH

#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA

#include "cuda_runtime.h"

#else

#include <thrust/detail/config.h>


typedef struct float3
{
  float x, y, z;
} float3;

typedef struct float4
{
  float x, y, z, w;
} float4;

struct uint3
{
  unsigned int x, y, z;
};


static __inline__ __host__ __device__ float3 make_float3(float x, float y, float z)
{
  float3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __host__ __device__ float4 make_float4(float x, float y, float z, float w)
{
  float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __host__ __device__ uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
  uint3 t; t.x = x; t.y = y; t.z = z; return t;
}

#endif


static __inline__ __host__ __device__ float3 make_float3(float4 a)
{
  float3 t; t.x = a.x; t.y = a.y; t.z = a.z; return t;
}

static __inline__ __host__ __device__ float4 make_float4(float3 a, float w)
{
  float4 t; t.x = a.x; t.y = a.y; t.z = a.z; t.w = w; return t;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}


inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}


inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}


inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}


inline __host__ __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}


inline __host__ __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}


inline __host__ __device__ float3 normalize(float3 v)
{
    return ((1.0f / sqrt(dot(v,v))) * v);
}
inline __host__ __device__ float4 normalize(float4 v)
{
    return ((1.0f / sqrt(dot(v,v))) * v);
}


inline __host__ __device__ float cosAngle(float3& a_vector0, float3 a_vector1)
{
    // compute length of vectors
    double n0 = sqrt(dot(a_vector0, a_vector0));
    double n1 = sqrt(dot(a_vector1, a_vector1));
    double val = n0 * n1;

    // check if lengths of vectors are not zero
    if (fabs(val) < 0.00001)
    {
        return (0);
    }

    // compute angle
    return(dot(a_vector0, a_vector1)/(val));
}


inline __host__ __device__ float4 matrixMul(float* r, float4 v)
{
    return make_float4(r[0]*v.x + r[1]*v.y + r[2]*v.z + r[3]*v.w, r[4]*v.x + r[5]*v.y + r[6]*v.z +r[7]*v.w, r[8]*v.x + r[9]*v.y + r[10]*v.z + r[11]*v.w, r[12]*v.x + r[13]*v.y + r[14]*v.z + r[15]*v.w);
}


inline __host__ __device__ float3 matrixMul(float* r, float3 v)
{
    return make_float3(r[0]*v.x + r[1]*v.y + r[2]*v.z, r[4]*v.x + r[5]*v.y + r[6]*v.z, r[8]*v.x + r[9]*v.y + r[10]*v.z);
}


inline __host__ __device__ float* matrixMul(float* a, float* b)
{
    float* c = new float[16];
    c[0] = a[0]*b[0]+a[1]*b[4]+a[2]*b[8]+a[3]*b[12];
    c[1] = a[0]*b[1]+a[1]*b[5]+a[2]*b[9]+a[3]*b[13];
    c[2] = a[0]*b[2]+a[1]*b[6]+a[2]*b[10]+a[3]*b[14];
    c[3] = a[0]*b[3]+a[1]*b[7]+a[2]*b[11]+a[3]*b[15];

    c[4] = a[4]*b[0]+a[5]*b[4]+a[6]*b[8]+a[7]*b[12];
    c[5] = a[4]*b[1]+a[5]*b[5]+a[6]*b[9]+a[7]*b[13];
    c[6] = a[4]*b[2]+a[5]*b[6]+a[6]*b[10]+a[7]*b[14];
    c[7] = a[4]*b[3]+a[5]*b[7]+a[6]*b[11]+a[7]*b[15];

    c[8] = a[10]*b[8]+a[11]*b[12]+a[8]*b[0]+a[9]*b[4];
    c[9] = a[10]*b[9]+a[11]*b[13]+a[8]*b[1]+a[9]*b[5];
    c[10] = a[10]*b[10]+a[11]*b[14]+a[8]*b[2]+a[9]*b[6];
    c[11] = a[10]*b[11]+a[11]*b[15]+a[8]*b[3]+a[9]*b[7];

    c[12] = a[12]*b[0]+a[13]*b[4]+a[14]*b[8]+a[15]*b[12];
    c[13] = a[12]*b[1]+a[13]*b[5]+a[14]*b[9]+a[15]*b[13];
    c[14] = a[12]*b[2]+a[13]*b[6]+a[14]*b[10]+a[15]*b[14];
    c[15] = a[12]*b[3]+a[13]*b[7]+a[14]*b[11]+a[15]*b[15];

    return c;
}

#endif
