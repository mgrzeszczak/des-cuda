#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

#include "c_utils.h"
#include "des.h"
#include "des_utils.h"
#include "bit_utils.h"
#include "des_consts.h"
#include "des_kernel.h"

int main(int argc, char** argv) {
	//uint64_t key = 0x133457799BBCDFF1;
	uint64_t key = 0x133457799BBCDFF0;
	permutations* perm = create_permutations();

	uint64_t gen_key = des_generate_key();
	bits_print_grouped(gen_key, 8, 64);
	printf("\n");
	return 0;

	uint64_t *blocks;
	int block_count;
	chop_into_blocks("12345678", 8, &blocks, &block_count);
	printf("Size = %d\n", block_count);
	bits_print_grouped(blocks[0], 8, 64);
	printf("\n");
	free(blocks);

	bits_print_grouped(key, 8, 64);
	printf("\n");
	uint64_t keys[16];
	des_create_subkeys(key, keys, perm);
	uint64_t block = 0x0123456789ABCDEF;
	uint64_t result = encode_block(block, keys, perm);
	bits_print_grouped(result, 8, 64);
	printf("\n");

	free_permutations(perm);
	return EXIT_SUCCESS;
}