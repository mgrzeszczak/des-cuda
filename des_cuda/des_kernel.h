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
__global__ void cuda_crack_des_kernel(uint64_t block, uint64_t encoded,uint64_t limit, uint64_t *key, int *done);

void run_des_crack(uint64_t block, uint64_t encoded, int key_length, uint64_t *key);
void run_des_encode_block(uint64_t key, uint64_t block, uint64_t *result);
uint64_t calculate_limit(int key_length);

__global__ void cuda_des_encode_block(uint64_t block, uint64_t key, uint64_t *encoded) {
	uint64_t keys[16];
	des_create_subkeys(key, keys);
	uint64_t result = des_encode_block(block, keys);
	*encoded = result;
}

__constant__ uint64_t POW_2_42 = 4398046511104;
__constant__ uint16_t POW_2_14 = 16384;

__global__ void cuda_crack_des_kernel(uint64_t block, uint64_t encoded,uint64_t limit, uint64_t *key, int* done) {
	const int threadCount = 512;
	int blockCount = gridDim.x;
	int tbid = blockIdx.x*threadCount + threadIdx.x;
	int id = threadIdx.x;
	int offset = blockIdx.x * threadCount;

	if ((*done) == 1){
		return;
	}

	// calculate current thread starting position
	uint64_t current_key = 0;
	uint64_t total_id = (uint64_t)blockIdx.x*(uint64_t)threadCount + (uint64_t)threadIdx.x;
	bits_copy(total_id, &current_key, 0, 1, 7);
	bits_copy(total_id, &current_key, 7, 9, 7);
	bits_copy(total_id, &current_key, 14, 17, 7);
	//bits_copy(total_id, &current_key, 21, 25, 1);
	//bits_copy(total_id, &current_key, 22, 26, 1);
	//bits_copy(total_id, &current_key, 23, 27, 1);

	/*if (tbid == 16383) {
		*key = current_key;
	}*/
	
	uint64_t result;
	
	//const uint64_t max = 34359738368;
    	for (uint64_t i=0;i<limit;i++) {
    //for (uint64_t i = 0; i < max; i++) {
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
			*done = 1;
			break;
		}

	        if (i % 1024 == 0 && (*done)==1) {
	        	break;
	        }
	}
}

uint64_t calculate_limit(int key_length) {
	const int offset = 24;
	int input = key_length;
	input -= offset;
	int z = (input - 1) / 8 + 1;
	input -= z;

	uint64_t limit = 1;
	for (int i = 0; i < input; i++) {
		limit <<= 1;
	}
	//printf("%d -> %d, %u\n", key_length, input,limit);
	return limit;
}

void run_des_crack(uint64_t block, uint64_t encoded, int key_length, uint64_t *key) {
	uint64_t *dev_key;
	uint64_t key_val = 0;
	int *done;
	uint64_t limit = calculate_limit(key_length);
	
	// select device
	//_cudaSetDevice(0);	
	//_cudaResizeStack();
	// allocate memory
	_cudaMalloc((void**)&dev_key, sizeof(uint64_t));
	_cudaMalloc((void**)&done,sizeof(int));
	// copy values
	_cudaMemcpy(dev_key, &key_val, sizeof(uint64_t), cudaMemcpyHostToDevice);
	int done_value = 0;
	_cudaMemcpy(done,&done_value,sizeof(int),cudaMemcpyHostToDevice);

	//cuda_crack_des_kernel << <32678, 512 >> >(block, encoded, limit, dev_key, done);
	cuda_crack_des_kernel << <4096, 512 >> >(block, encoded, limit, dev_key, done);
	_cudaDeviceSynchronize("crack_des_kernel");

	//_cudaMemcpy(&done_value,done,sizeof(int),cudaMemcpyDeviceToHost);
	//printf("done = %d\n",done_value);

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
