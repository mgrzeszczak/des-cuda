#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include "consts.h"

void _cudaSetDevice(int device) {
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		ERR("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
	}
}

void _cudaMalloc(void **dest, size_t size) {
	cudaError_t cudaStatus = cudaMalloc(dest, size);
	if (cudaStatus != cudaSuccess) {
		ERR("cudaMalloc failed!\n");
	}
}

void _cudaMemset(void *dest, int val, size_t size) {
	cudaError_t cudaStatus = cudaMemset(dest, 0, size);
	if (cudaStatus != cudaSuccess) {
		ERR("cudaMemset failed!\n");
	}
}

void _cudaMemcpy(void *dest, const void *src, size_t size, cudaMemcpyKind kind) {
	cudaError_t cudaStatus = cudaMemcpy(dest, src, size, kind);
	if (cudaStatus != cudaSuccess) {
		ERR("cudaMemcpy failed!\n");
	}
}

void _cudaDeviceSynchronize(char* s) {
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		ERR("%s launch failed: %s\n",s, cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		ERR("cudaDeviceSynchronize returned error code %d after launching %s!\n",cudaStatus,s);
	}
}

void _cudaPrintMemory() {
	//if (!DEBUG) return;
	size_t mem_free;
	size_t mem_total;
	cudaMemGetInfo(&mem_free, &mem_total);
	mem_free /= MB;
	mem_total /= MB;
	printf("MEMORY: \nfree %u MB\ntotal %u MB\n", mem_free, mem_total);
}
