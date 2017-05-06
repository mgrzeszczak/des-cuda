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
	uint64_t key = 0x00000000005efefe;
	uint64_t block = 0x0123456789ABCDEF;
	uint64_t encoded = full_des_encode_block(key, block);

	printf("Real key:\n");
	bits_print_grouped(key, 8, 64);
	printf("Cracking...\n");
	uint64_t cracked_key;
	run_des_crack(block, encoded, &cracked_key);
	printf("Cracked key:\n");
	bits_print_grouped(cracked_key, 8, 64);

	/*
	for (uint16_t i = 0; i < 128;i++) {
		bits_copy(i, &key, 0, 1, 7);
		bits_print_grouped(key, 8, 16);
		key = 0;
	}*/

	/*
	for (uint16_t i = 0; i < POW_2_14; i++) {
		bits_copy(i, &key, 0, 1, 7);
		bits_copy(i, &key, 7, 9, 7);
		bits_print_grouped(key, 8, 16);
		key = 0;
	}*/

	/*uint64_t block = 0x0123456789ABCDEF;

	for (uint64_t i = 0; i < POW_2_42; i++) {
		key = 0;
		bits_copy(i, &key, 0, 17, 7);
		bits_copy(i, &key, 7, 25, 7);
		bits_copy(i, &key, 14, 33, 7);
		bits_copy(i, &key, 21, 41, 7);
		bits_copy(i, &key, 28, 49, 7);
		bits_copy(i, &key, 35, 57, 7);

		bits_print_grouped(key, 8, 64);
	}*/





	

	/*uint64_t key = 0x133457799BBCDFF1;
	uint64_t block = 0x0123456789ABCDEF;
	bits_print_grouped(key, 8, 64);
	printf("\n");
	bits_print_grouped(0x133457799BBCDFF0, 8, 64);
	bits_print_grouped(0x133457799BBCDFF1, 8, 64);
	bits_print_grouped(0x133457799BBCDEF0, 8, 64);*/
	

	/*
	uint64_t block = 0x0123456789ABCDEF;
	uint64_t encoded = host_des_encode_block(key, block);

	bits_print_grouped(encoded, 8, 64);
	printf("\n");
	key = 0x133457799BBCDFF0;
	encoded = host_des_encode_block(key, block);
	bits_print_grouped(encoded, 8, 64);
	printf("\n");

	uint64_t cracked_key;
	run_des_crack(block, encoded, &cracked_key);

	bits_print_grouped(cracked_key, 8, 64);*/

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
