#include "matrix.h"
#include "matrix_math.h"
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#define INT 2/1024
#define ROWS INT*1024 
#define COLS INT*1024 
#define MU 0.0
#define SIGMA 1
#define REQUIRES_GRAD 0 


int main(){

	// MATRIX_TIMER(matrix* m = matrix_random_normal(ROWS, COLS, 0, 1, 1));
	matrix* m = matrix_alloc(10, 10, 0);
	// matrix* n = matrix_ones(10, 10, 0);
	matrix_print(m);
	// matrix* o = matrix_alloc(10, 10, 0);
	// printf("%s\n", get_optype_string(ADD));
	// printf("%s\n", get_optype_string(MATMUL));
	


	return 0;
}
