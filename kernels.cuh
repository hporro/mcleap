#pragma once

#include "Common_Triangulation.h"

// Functor is supposed to output 0 if not and 1 if yes
template<class Functor, typename... Args>
__global__ void resCollector(int n, int* res, Functor f, Args... args) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		res[i] = f(i,args...);
	}
}

//flag stores how many threads actually do work
__global__ void compactIncScanned(int n, int* scanned, int* compacted, int* flag) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i == 0) {
		if (scanned[0] == 1)compacted[0] = 0;
		flag[0] = scanned[n - 1];
	}
	if (i < n && i>0) {
		if (scanned[i] > scanned[i - 1]) {
			compacted[scanned[i]-1] = i;
		}
	}
}

template<class Functor, typename... Args>
__global__ void doGivenCompacted(int* compacted, int* flag, Functor f , Args... args) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < flag[0]) {
		f(compacted[i], args...);
	}
}

__global__ void move_points_kernel(int n, glm::vec2* pos, glm::vec2* d) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		pos[i]+=d[i];
	}
}