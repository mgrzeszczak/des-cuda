#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "des.h"
#include "bit_utils.h"
#include "des_consts.h"

uint64_t bits_permutate_x(uint64_t key, int*permutation, int length, int key_length) {
	uint64_t result = 0;
	for (int i = 0; i < length; i++) {
		bits_set(&result, i, bits_bit64(key, key_length - permutation[length - 1 - i]));
	}
	return result;
}

int main(int argc, char** argv) {
	uint64_t key = 0x133457799BBCDFF1;
	uint64_t keys[16];
	des_create_subkeys(key,keys);
	
	for (int i = 0; i < 16; i++) {
		bits_print_grouped(keys[i], 6, 48);
		printf("\n");
	}
	return EXIT_SUCCESS;
}