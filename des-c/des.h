#pragma once
#include <stdint.h>
#include <stdio.h>
#include "des_consts.h"
#include "bit_utils.h"

void des_create_subkeys(uint64_t key, uint64_t *keys);

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