#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

__kernel void _copyZ(__global const float* z0, __global float* z1, __global float* z2, __global float* z3, __global float* z4,
		__global float* z5, __global float* z6, __global float* z7, __global float* z8, __global float* z9, __global float* z10,
		__global float* z11, __global float* z12, unsigned int N) {
	for (unsigned int i = get_global_id(0); i < N; i += get_global_size(0)) {
		float v = z0[i];
		z1[i] = v, z2[i] = v, z3[i] = v, z4[i] = v, z5[i] = v, z6[i] = v;
		z7[i] = v, z8[i] = v, z9[i] = v, z10[i] = v, z11[i] = v, z12[i] = v;
	}
}

__kernel void _updateZC(__global const float* x, __global const float* p, __global float* z, float ri, unsigned int N) {
	for (unsigned int i = get_global_id(0); i < N; i += get_global_size(0)) {
		z[i] = max(0.0F, x[i] + p[i] / ri);
	}
}

__kernel void _updateZB(__global const float* x, __global const float* p, __global float* z, float ri, float bgr,
		unsigned int N) {
	for (unsigned int i = get_global_id(0); i < N; i += get_global_size(0)) {
		float a = x[i] + p[i] / ri;
		if (fabs(a) <= bgr) z[i] = 0;
		else if (a > bgr) z[i] = a - bgr;
		else z[i] = a + bgr;
	}
}
