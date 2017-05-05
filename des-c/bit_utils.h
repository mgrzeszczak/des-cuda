#pragma once
#include <stdint.h>

int bits_bit8(uint8_t input, int nr);
int bits_bit16(uint16_t input, int nr);
int bits_bit32(uint32_t input, int nr);
int bits_bit64(uint64_t input, int nr);
void bits_split64(uint64_t input, uint64_t *left, uint64_t *right);
void bits_split(uint64_t input, uint64_t *left, uint64_t *right, int len);
void bits_set(uint64_t input, int nr, uint64_t val);
void bits_copy(uint64_t src, uint64_t *dst, int offset, int len);
void bits_print(uint64_t val);
uint64_t bits_cycle_left(uint64_t val, int shift, int size);
uint64_t bits_cycle_right(uint64_t val, int shift, int size);
uint64_t bits_permutate(uint64_t key, int*permutation, int length);
uint64_t bits_key_permutation(uint64_t key, int* permutation);

int bits_bit8(uint8_t input, int nr) {
	return (input >> nr) & 1;
}
int bits_bit16(uint16_t input, int nr) {
	return (input >> nr) & 1;
}
int bits_bit32(uint32_t input, int nr) {
	return (input >> nr) & 1;
}
int bits_bit64(uint64_t input, int nr) {
	return (input >> nr) & 1;
}

void bits_set(uint64_t *data, int nr, uint64_t val) {
	*data = *data | ((uint64_t)val << nr);
}

void bits_copy(uint64_t src, uint64_t *dst, int src_offset,int dst_offset, int len) {
	for (int i = 0; i < len; i++) {
		bits_set(dst, i+dst_offset, bits_bit64(src, src_offset + i));
	}
}

void bits_split64(uint64_t input, uint64_t *left, uint64_t *right) {
	bits_split(input, left, right, 64);
}

void bits_split(uint64_t input, uint64_t *left, uint64_t *right, int len) {
	*left = *right = 0;
	bits_copy(input, right, 0,0, len / 2);
	bits_copy(input, left, len / 2,0, len / 2);
}

void bits_print(uint64_t val) {
	for (int i = 63; i >=0 ; i--) {
		printf("%d ", bits_bit64(val, i));
		if (i % 8 == 0) printf("  ");
	}
	printf("\n\n");
}

uint64_t bits_cycle_left(uint64_t val, int shift, int size) {
	uint64_t tmp = 0;
	bits_copy(val, &tmp, size-shift, 0, shift);
	val = val << shift;
	bits_copy(tmp, &val, 0, 0, shift);
	val = val & (~(0xffffffffffffffff << size));
	return val;
}

uint64_t bits_cycle_right(uint64_t val, int shift, int size) {
	uint64_t tmp = 0;
	bits_copy(val, &tmp, size - shift, 0, shift);
	val = val << shift;
	bits_copy(tmp, &val, 0, 0, shift);
	val = val & (~(0xffffffffffffffff << size));
	return val;
}

uint64_t bits_key_permutation(uint64_t key, int* permutation) {
	uint64_t result = 0;
	for (int i = 0; i < 56; i++) {
		//printf("Settings %d based on %d\n", i, 64-permutation[56 - i - 1]);
		bits_set(&result, i, bits_bit64(key, 64-permutation[56-1-i]));
		//bits_print(result);
	}
	return result;
}

uint64_t bits_permutate(uint64_t key, int*permutation, int length) {
	uint64_t result = 0;
	for (int i = 0; i < length; i++) {
		bits_set(&result, i, bits_bit64(key, 64 - permutation[length - 1 - i]));
	}
	return result;
}