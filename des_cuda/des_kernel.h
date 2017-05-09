#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include "cuda_utils.h"
#include "bit_utils.h"
#include "c_utils.h"
#include "consts.h"
#include "des.h"

__global__ void cuda_des_encode_block(uint64_t block, uint64_t key, uint64_t *encoded);
__global__ void cuda_crack_des_kernel(uint64_t block, uint64_t encoded, uint64_t *key);

void run_des_crack(uint64_t block, uint64_t encoded, uint64_t *key);
void run_des_encode_block(uint64_t key, uint64_t block, uint64_t *result);

__global__ void cuda_des_encode_block(uint64_t block, uint64_t key, uint64_t *encoded) {
	uint64_t keys[16];
	des_create_subkeys(key, keys);
	uint64_t result = des_encode_block(block, keys);
	*encoded = result;
}

__constant__ uint64_t POW_2_42 = 4398046511104;
__constant__ uint16_t POW_2_14 = 16384;

__global__ void cuda_crack_des_kernel(uint64_t block, uint64_t encoded, uint64_t *key) {
	const int threadCount = 512;
	int blockCount = gridDim.x;
	int tbid = blockIdx.x*threadCount + threadIdx.x;
	int id = threadIdx.x;
	int offset = blockIdx.x * threadCount;

	// calculate current thread starting position
	uint64_t current_key = 0;
	uint64_t total_id = (uint64_t)blockIdx.x*(uint64_t)threadCount + (uint64_t)threadIdx.x;
	bits_copy(total_id, &current_key, 0, 1, 7);
	bits_copy(total_id, &current_key, 7, 9, 7);
	bits_copy(total_id, &current_key, 14, 17, 7);

	/*if (tbid == 16383) {
		*key = current_key;
	}*/
	
	uint64_t result;
	
	const uint64_t max = 34359738368;
    //for (uint64_t i=0;i<10;i++) {
    for (uint64_t i = 0; i < max; i++) {
		// clear first 40 bits
		current_key = current_key << 40;
		current_key = current_key >> 40;
		// copy 
		bits_copy(i, &current_key, 0, 25, 7);
		bits_copy(i, &current_key, 7, 33, 7);
		bits_copy(i, &current_key, 14, 41, 7);
		bits_copy(i, &current_key, 21, 49, 7);
		bits_copy(i, &current_key, 28, 57, 7);

		result = full_des_encode_block(current_key, block);
		if (result == encoded) {
			*key = current_key;
			return;
		}

        if ((*key!=0)){
            return;
        }
	
		//if (i % 1024 == 0) {
		//if ((*flag) == true) {
		//	return;
		//}
		//}
	}
}

void run_des_crack(uint64_t block, uint64_t encoded, uint64_t *key) {
	uint64_t *dev_key;
	uint64_t key_val = 0;
	// select device
	//_cudaSetDevice(0);	
	//_cudaResizeStack();
	// allocate memory
	_cudaMalloc((void**)&dev_key, sizeof(uint64_t));
	// copy values
	_cudaMemcpy(dev_key, &key_val, sizeof(uint64_t), cudaMemcpyHostToDevice);

	cuda_crack_des_kernel << <4096, 512 >> >(block, encoded, dev_key);
	_cudaDeviceSynchronize("crack_des_kernel");

	// copy result
	_cudaMemcpy(key, dev_key, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	// free memory
	cudaFree(dev_key);
}

void run_des_encode_block(uint64_t key, uint64_t block, uint64_t *result) {
	uint64_t *dev_result;
	//_cudaSetDevice(0);
	_cudaMalloc((void**)&dev_result, sizeof(uint64_t));

	cuda_des_encode_block<<<1,1>>>(block, key, dev_result);
	_cudaDeviceSynchronize("cuda_des_encode_block");

	_cudaMemcpy(result, dev_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaFree(dev_result);
}
