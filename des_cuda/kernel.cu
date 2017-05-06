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
	uint64_t key = 0x133457799BBCDFF1;
	uint64_t block = 0x0123456789ABCDEF;
	uint64_t encoded = host_des_encode_block(key, block);

	uint64_t cracked_key;
	run_des_crack(block, encoded, &cracked_key);

	bits_print_grouped(cracked_key, 8, 64);

	/*

	uint64_t *blocks;
	int block_count;
	chop_into_blocks("12345678", 8, &blocks, &block_count);
	

	uint64_t key = 0x133457799BBCDFF1;
	uint64_t block = 0x0123456789ABCDEF;
	uint64_t result;

	run_des_encode_block(key, blocks[0], &result);
	bits_print_grouped(result, 8, 64);
	printf("\n");

	uint64_t keys[16];
	des_create_subkeys(key, keys);
	uint64_t cpu_result = des_encode_block(blocks[0], keys);
	bits_print_grouped(result, 8, 64);
	printf("\n");

	//checked at http://des.online-domain-tools.com/
	//bits_print_grouped(0x85e813540f0ab405, 8, 64);
	bits_print_grouped(0x8b96b79529cca218, 8, 64);
	printf("\n");

	free(blocks);*/
	return EXIT_SUCCESS;
}