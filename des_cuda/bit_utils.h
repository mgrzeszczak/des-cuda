#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>

__device__ __host__ uint64_t bits_bit8(uint8_t input, int nr);
__device__ __host__ uint64_t bits_bit16(uint16_t input, int nr);
__device__ __host__ uint64_t bits_bit32(uint32_t input, int nr);
__device__ __host__ uint64_t bits_bit64(uint64_t input, int nr);

__device__ __host__ void bits_split64(uint64_t input, uint64_t *left, uint64_t *right);
__device__ __host__ void bits_split(uint64_t input, uint64_t *left, uint64_t *right, int len);
__device__ __host__ void bits_set(uint64_t input, int nr, uint64_t val);
__device__ __host__ void bits_copy(uint64_t src, uint64_t *dst, int offset, int len);
__device__ __host__ void bits_print(uint64_t val);
__device__ __host__ void bits_print_grouped(uint64_t val, int group_size, int length);

__device__ __host__ uint64_t bits_cycle_left(uint64_t val, int shift, int size);
__device__ __host__ uint64_t bits_cycle_right(uint64_t val, int shift, int size);
__device__ __host__ uint64_t bits_permutate(uint64_t key, int*permutation, int length, int key_length);

__device__ __host__ uint64_t bits_bit8(uint8_t input, int nr) {
	return (input >> nr) & 1;
}
__device__ __host__ uint64_t bits_bit16(uint16_t input, int nr) {
	return (input >> nr) & 1;
}
__device__ __host__ uint64_t bits_bit32(uint32_t input, int nr) {
	return (input >> nr) & 1;
}
__device__ __host__ uint64_t bits_bit64(uint64_t input, int nr) {
	return (input >> nr) & 1;
}

__device__ __host__ void bits_set(uint64_t *data, int nr, uint64_t val) {
	*data = *data | ((uint64_t)val << nr);
}

__device__ __host__ void bits_copy(uint64_t src, uint64_t *dst, int src_offset,int dst_offset, int len) {
	for (int i = 0; i < len; i++) {
		bits_set(dst, i+dst_offset, bits_bit64(src, src_offset + i));
	}
}

__device__ __host__ void bits_split64(uint64_t input, uint64_t *left, uint64_t *right) {
	bits_split(input, left, right, 64);
}

__device__ __host__ void bits_split(uint64_t input, uint64_t *left, uint64_t *right, int len) {
	*left = *right = 0;
	bits_copy(input, right, 0,0, len / 2);
	bits_copy(input, left, len / 2,0, len / 2);
}

__device__ __host__ void bits_print(uint64_t val) {
	for (int i = 63; i >=0 ; i--) {
		printf("%d ", bits_bit64(val, i));
		if (i % 8 == 0) printf("  ");
	}
}

__device__ __host__ uint64_t bits_cycle_left(uint64_t val, int shift, int size) {
	uint64_t tmp = 0;
	bits_copy(val, &tmp, size-shift, 0, shift);
	val = val << shift;
	bits_copy(tmp, &val, 0, 0, shift);
	val = val & (~(0xffffffffffffffff << size));
	return val;
}

__device__ __host__ uint64_t bits_cycle_right(uint64_t val, int shift, int size) {
	uint64_t tmp = 0;
	bits_copy(val, &tmp, size - shift, 0, shift);
	val = val << shift;
	bits_copy(tmp, &val, 0, 0, shift);
	val = val & (~(0xffffffffffffffff << size));
	return val;
}

__device__ __host__ uint64_t bits_permutate(uint64_t key,int*permutation, int length, int key_length) {
	uint64_t result = 0;
	for (int i = 0; i < length; i++) {
		bits_set(&result, i, bits_bit64(key, key_length - permutation[length - 1 - i]));
	}
	return result;
}

__device__ __host__ void bits_print_grouped(uint64_t val, int group_size, int length) {
	for (int i = length - 1; i >= 0;i--) {
		printf("%d",bits_bit64(val,i));
		if (i%group_size == 0) {
			printf(" ");
		}
	}
}