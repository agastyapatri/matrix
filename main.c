#include "matrix.h"
#include "matrix_math.h"
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#define INT 5
#define ROWS INT*1024 
#define COLS INT*1024 
#define MU 0.0
#define SIGMA 1
#define REQUIRES_GRAD 0 


int main(){

	// MATRIX_TIMER(matrix* m = matrix_random_normal(ROWS, COLS, 0, 1, 1));
	matrix* m = matrix_ones(ROWS, COLS, 0);
	matrix* n = matrix_ones(COLS, ROWS, 0);
	matrix* o = matrix_alloc(ROWS, ROWS, 0);

	// MATRIX_TIMER(MATRIX_MATMUL(m, n, o));
	// MATRIX_TIMER(matrix_matmul(m, n));
	// matrix_print(o);


	return 0;
}
