#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "des_consts.h"
#include "bit_utils.h"
#include "c_utils.h"
#include <string.h>

uint64_t des_generate_key();
void chop_into_blocks(char* data, int length, uint64_t **blocks, int *out_block_count);

uint64_t des_generate_key() {
	srand(time(0));
	uint64_t key = 0;
	for (int i = 0; i < 64; i++) {
		bits_set(&key, i, rand() % 64>32);
	}
	return key;
}

void chop_into_blocks(char* data, int length, uint64_t **blocks, int *out_block_count) {
	double d = length / 8.0;
	int block_count = (int)(ceil(d));
	char* filled = (char*)_malloc(block_count * 8);
	memset(filled, 0, block_count * 8);
	memcpy(filled, data, length);

	*blocks = (uint64_t*)_malloc(sizeof(uint64_t)*block_count);
	int offset = 0;
	for (int i = 0; i < block_count; i++) {
		*blocks[i] = ((uint64_t)data[0 + i*offset]) << 56 |
			((uint64_t)data[1 + i*offset]) << 48 |
			((uint64_t)data[2 + i*offset]) << 40 |
			((uint64_t)data[3 + i*offset]) << 32 |
			((uint64_t)data[4 + i*offset]) << 24 |
			((uint64_t)data[5 + i*offset]) << 16 |
			((uint64_t)data[6 + i*offset]) << 8 |
			((uint64_t)data[7 + i*offset]) << 0;
		offset += 8;
	}
	*out_block_count = block_count;
}