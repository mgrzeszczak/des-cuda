#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "des.h"
#include "bit_utils.h"
#include "des_consts.h"


int main(int argc, char** argv) {
	uint64_t key = 0x133457799BBCDFF1;
	uint64_t keys[16];
	des_create_subkeys(key,keys);
	uint64_t block = 0x0123456789ABCDEF;
	uint64_t result = encode_block(block,keys);
	bits_print_grouped(result, 8, 64);
	printf("\n");
	return EXIT_SUCCESS;
}