#pragma once

#include <stdint.h>
#include "des_consts.h"
#include "bit_utils.h"
#include "c_utils.h"

typedef struct permutations {
	int* PC_1;
	int* PC_2;
	int* IP;
	int* E_BIT;
	int* S1;
	int* S2;
	int* S3;
	int* S4;
	int* S5;
	int* S6;
	int* S7;
	int* S8;
	int** S;
	int* P;
	int* IP_REV;
	int* SHIFTS;
} permutations;


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

permutations* create_permutations() {
	permutations *perms = (permutations*)_malloc(sizeof(permutations));
	perms->E_BIT = E_BIT;
	perms->IP = IP;
	perms->IP_REV = IP_REV;
	perms->P = P;
	perms->PC_1 = PC_1;
	perms->PC_2 = PC_2;
	perms->SHIFTS = SHIFTS;
	perms->S1 = S1;
	perms->S2 = S2;
	perms->S3 = S3;
	perms->S4 = S4;
	perms->S5 = S5;
	perms->S6 = S6;
	perms->S7 = S7;
	perms->S8 = S8;
	perms->S = (int**)_malloc(sizeof(int*) * 8);
	perms->S[0] = S1;
	perms->S[1] = S2;
	perms->S[2] = S3;
	perms->S[3] = S4;
	perms->S[4] = S5;
	perms->S[5] = S6;
	perms->S[6] = S7;
	perms->S[7] = S8;
	return perms;
}

void free_permutations(permutations *perm) {
	free(perm->S);
	free(perm);
}