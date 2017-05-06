#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include "cuda_utils.h"
#include "c_utils.h"
#include "consts.h"
#include "des.h"

__global__ void cuda_des_encode_block(uint64_t block, uint64_t key, uint64_t *encoded);
__global__ void cuda_crack_des_kernel(uint64_t block, uint64_t encoded, bool* flag, uint64_t *key);

void run_des_crack(uint64_t block, uint64_t encoded, uint64_t *key);
void run_des_encode_block(uint64_t key, uint64_t block, uint64_t *result);

__global__ void cuda_des_encode_block(uint64_t block, uint64_t key, uint64_t *encoded) {
	uint64_t keys[16];
	des_create_subkeys(key, keys);
	uint64_t result = des_encode_block(block, keys);
	*encoded = result;
}

__global__ void cuda_crack_des_kernel(uint64_t block, uint64_t encoded, bool* flag, uint64_t *key) {
	const int threadCount = 1024;
	int blockCount = gridDim.x;
	int tbid = blockIdx.x*threadCount + threadIdx.x;
	int id = threadIdx.x;
	int offset = blockIdx.x * threadCount;

	if (tbid == 0) {
		*key = 1;
		*key = (*key)<<63;
	}
}

void run_des_crack(uint64_t block, uint64_t encoded, uint64_t *key) {
	uint64_t *dev_key;
	bool *dev_flag;
	bool flag_value = false;
	// select device
	_cudaSetDevice(0);	
	// allocate memory
	_cudaMalloc((void**)&dev_key, sizeof(uint64_t));
	_cudaMalloc((void**)&dev_flag, sizeof(bool));
	// copy values
	_cudaMemcpy(dev_flag, &flag_value, sizeof(bool), cudaMemcpyHostToDevice);

	cuda_crack_des_kernel << <1, 1024 >> >(block, encoded, dev_flag, dev_key);
	_cudaDeviceSynchronize("crack_des_kernel");

	// copy result
	_cudaMemcpy(key, dev_key, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	// free memory
	cudaFree(dev_key);
	cudaFree(dev_flag);
}

void run_des_encode_block(uint64_t key, uint64_t block, uint64_t *result) {
	uint64_t *dev_result;
	_cudaSetDevice(0);
	_cudaMalloc((void**)&dev_result, sizeof(uint64_t));

	cuda_des_encode_block<<<1,1>>>(block, key, dev_result);
	_cudaDeviceSynchronize("cuda_des_encode_block");

	_cudaMemcpy(result, dev_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaFree(dev_result);
}