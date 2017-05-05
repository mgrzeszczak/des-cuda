#pragma once
#include <stdint.h>
#include <stdio.h>
#include "des_consts.h"
#include "bit_utils.h"

void des_create_subkeys(uint64_t key, uint64_t *keys);
int64_t encode_block(uint64_t block, uint64_t *keys);
int64_t f(uint64_t right, uint64_t key);

void des_create_subkeys(uint64_t key, uint64_t *keys) {
	uint64_t key_plus = bits_permutate(key, PC_1, 56, 64);	

	uint64_t left, right;
	bits_split(key_plus, &left, &right, 56);

	uint64_t c_blocks[17];
	uint64_t d_blocks[17];

	c_blocks[0] = left;
	d_blocks[0] = right;

	for (int i = 1; i <= 16; i++) {
		c_blocks[i] = bits_cycle_left(c_blocks[i - 1], SHIFTS[i - 1], 28);
		d_blocks[i] = bits_cycle_left(d_blocks[i - 1], SHIFTS[i - 1], 28);
	}

	for (int i = 1; i <= 16; i++) {
		keys[i - 1] = c_blocks[i] << 28 | d_blocks[i];
		keys[i-1] = bits_permutate(keys[i - 1], PC_2, 48, 56);
	}
}

int64_t f(uint64_t right, uint64_t key) {
	uint64_t expanded = bits_permutate(right, E_BIT, 48, 32);
	uint64_t xor = expanded ^ key;
	uint64_t b[8];
	uint64_t s[8];

	for (int i = 7; i >=0; i--) {
		b[7-i] = 0;
 		bits_copy(xor, &b[7-i], i * 6, 0, 6);
		uint64_t bb = b[7 - i];

		uint64_t d_i = bits_bit64(bb, 5) << 1 | bits_bit64(bb, 0);
		uint64_t d_j = bits_bit64(bb, 4) << 3 |
			bits_bit64(bb, 3) << 2 |
			bits_bit64(bb, 2) << 1 |
			bits_bit64(bb, 1) << 0;

		int ii = d_i;
		int ij = d_j;
		int index = ii * 16 + ij;
		int data = S[7-i][index];
		s[7 - i] = data;
	}

	uint64_t s_res = s[0] << 28 |
		s[1] << 24 |
		s[2] << 20 |
		s[3] << 16 |
		s[4] << 12 |
		s[5] << 8 |
		s[6] << 4 |
		s[7] << 0;

	uint64_t p = bits_permutate(s_res, P, 32, 32);
	return p;
}

int64_t encode_block(uint64_t block, uint64_t *keys) {
	int64_t ip = bits_permutate(block, IP, 64, 64);
	int64_t left, right;
	bits_split64(ip, &left, &right);
	
	for (int i = 0; i < 16; i++) {
		// 16 iterations
		// n = i +1
		uint64_t prev_right = right;
		uint64_t prev_left = left;
		left = prev_right;
		right = prev_left ^ f(prev_right, keys[i]);
	}

	uint64_t reversed = right << 32 | left;
	uint64_t encoded = bits_permutate(reversed, IP_REV, 64, 64);
	return encoded;
}