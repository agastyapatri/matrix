#include "matrix.h"
#include "matrix_math.h"
#include "autograd.h"
#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define INT 1
#define ROWS INT*1024
#define COLS INT*1024
#define MU 0.0
#define SIGMA 1
#define REQUIRES_GRAD 0

int main(){
	srand(0);
	// matrix* m1 = matrix_random_uniform(6, 6, 0, 1, 1);
	// matrix* m2 = matrix_random_uniform(6, 6, 0, 1, 1);
	matrix* m1 = matrix_zeros(6, 6, 1);
	matrix* m2 = matrix_random_normal(6, 6, 1, 2, 1);
	matrix* m3 = matrix_std(m2);
	matrix_grad(m3);
	matrix* m2grad = matrix_from_raw(m2->grad, m2->rows, m2->cols);
	matrix_print(m2grad);
	return 0;
} 





