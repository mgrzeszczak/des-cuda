#pragma once
/******************************************
				MACROS
******************************************/
#define ERR(s,...) (fprintf(stderr, s,__VA_ARGS__),\
						exit(EXIT_FAILURE))
/******************************************
				CONSTANTS
******************************************/
const bool DEBUG = false;
const int KB = 1024;
const int MB = 1024 * 1024;
const int MAX_THREADS_PER_BLOCK = 1024;
const int MAX_BLOCKS_PER_GRID = 65535;