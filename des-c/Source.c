#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "bit_utils.h"

const int PC_1[56] = {
	57,49,41,33,25,17,9,
	1,58,50,42,34,26,18,
	10,2,59,51,43,35,27,
	19,11,3,60,52,44,36,
	63,55,47,39,31,23,15,
	7,62,54,46,38,30,22,
	14,6,61,53,45,37,29,
	21,13,5,28,20,12,4
};

const int PC_2[48] = {
	14,17,11,24,1,5,
	3,28,15,6,21,10,
	23,19,12,4,26,8,
	16,7,27,20,13,2,
	41,52,31,37,47,55,
	30,40,51,45,33,48,
	44,49,39,56,34,53,
	46,42,50,36,29,32 
};

int main(int argc, char** argv) {
	//uint64_t data = 0xff000000000000ff;
	//bits_print(data);

	uint64_t data = 0x0000000000000001;
	bits_print(data);
	data = bits_cycle_left(data, 1, 2);
	bits_print(data);
	data = bits_cycle_left(data, 1, 2);
	bits_print(data);
	data = bits_cycle_left(data, 1, 2);
	bits_print(data);
	data = bits_cycle_left(data, 1, 2);
	bits_print(data);
	return;
	/*
	for (int i = 0; i < 64; i++) {
		uint64_t shifted = bits_cycle_left(data, i);
		bits_print(shifted);
	}*/


	uint64_t key = 0x133457799BBCDFF1;
	bits_print(key);

	//uint64_t key_plus = bits_key_permutation(key, PC_1);
	uint64_t key_plus = bits_permutate(key, PC_1,56);
	bits_print(key_plus);

	uint64_t left, right;
	bits_split(key_plus, &left, &right, 56);
	printf("\nleft\n");
	bits_print(left);
	printf("\nright\n");
	bits_print(right);

	bits_print(left << 28 | right);

	return EXIT_SUCCESS;
}