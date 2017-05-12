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
#include "cuda_utils.h"


void parse_args(int argc, char** argv, int *key_length);
void usage(char* name);

void parse_args(int argc, char** argv, int *key_length){
	if (argc < 2){
		usage(argv[0]);
	}
	*key_length = atoi(argv[1]);
	if (*key_length <=0 || *key_length>64){
		usage(argv[0]);
	}
}

void usage(char* name){
	printf("Usage:\n %s key_length(1-64)\n",name);
	exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
	int key_length;
	parse_args(argc,argv,&key_length);
	printf("Key length: %d \n",key_length);
	uint64_t key = des_generate_key_length(key_length);
	uint64_t block = 0x0123456789ABCDEF;
	uint64_t encoded = full_des_encode_block(key, block);

	_cudaSetDevice(1);

	printf("Real key:\n");
	bits_print_grouped(key, 8, 64);
	printf("Cracking...\n");
	uint64_t cracked_key;

	clock_t start = clock();
	run_des_crack(block, encoded, key_length, &cracked_key);
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Time passed: %f s\n", seconds);

	printf("Cracked key:\n");
	bits_print_grouped(cracked_key, 8, 64);
	return EXIT_SUCCESS;
}
