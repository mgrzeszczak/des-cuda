#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include "cuda_utils.h"
#include "c_utils.h"
#include "consts.h"
#include "des.h"

__global__ void crack_des_kernel(uint64_t *blocks, uint64_t *encoded, permutations *perms, bool* flag, uint64_t *key);
uint64_t run_des_crack(uint64_t *blocks, uint64_t *encoded, int blockCount, permutations *perms);

__global__ void crack_des_kernel(uint64_t *blocks, uint64_t *encoded, permutations *perms, bool* flag, uint64_t *key) {
	*key = 1 << 63;
}

void run_des_crack(uint64_t *blocks, uint64_t *encoded, int blockCount, permutations *perms, uint64_t *key) {
	uint64_t *dev_blocks;
	uint64_t *dev_encoded;
	permutations *dev_permutations;

	uint64_t *dev_key;
	bool *dev_flag;

	_cudaSetDevice(0);

	_cudaMalloc((void**)&dev_blocks, sizeof(uint64_t)*blockCount);
	_cudaMalloc((void**)&dev_encoded, sizeof(uint64_t)*blockCount);
	_cudaMalloc((void**)&dev_permutations, sizeof(permutations));
	_cudaMalloc((void**)&dev_key, sizeof(uint64_t));
	_cudaMalloc((void**)&dev_flag, sizeof(bool));

	_cudaMemcpy(dev_blocks, blocks, sizeof(uint64_t)*blockCount,cudaMemcpyHostToDevice);
	_cudaMemcpy(dev_encoded, encoded, sizeof(uint64_t)*blockCount,cudaMemcpyHostToDevice);
	// TODO: memcpy permutations

	crack_des_kernel<<<1, 1>>>(dev_blocks,dev_encoded,dev_permutations,dev_flag,dev_key);
	_cudaDeviceSynchronize("crack_des_kernel");

	_cudaMemcpy(key, dev_key, sizeof(uint64_t), cudaMemcpyDeviceToHost);

	cudaFree(dev_blocks);
	cudaFree(dev_encoded);
	cudaFree(dev_permutations);
	cudaFree(dev_key);
	cudaFree(dev_flag);
}