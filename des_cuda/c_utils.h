#pragma once

#include "consts.h"

void* _malloc(int size) {
	void* mem = malloc(size);
	if (mem == NULL) ERR("Failed to malloc on CPU\n");
	return mem;
}

void* _realloc(void *data, int size) {	
	void* mem = realloc(data,size);
	if (mem == NULL) ERR("Failed to realloc ON CPU\n");
	return mem;
}

float randf() {
	return (float)rand() / (float)(RAND_MAX);
}