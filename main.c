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
	matrix* m1 = matrix_random_normal(10, 10, -1, 1, 1);
	matrix* m2 = matrix_random_normal(10, 10, 1, 2, 1);
	matrix* m3 = matrix_add(m1, m2);
	matrix* m4 = matrix_sin(m3);
	matrix* m5 = matrix_random_normal(10, 10, 0, 1, 1);
	matrix* m6 = matrix_matmul(m4, m2);
	matrix* m7 = matrix_sigmoid(m6);

	// matrix_print(m6);

	// matrix_backward(m7);
	// matrix* m6grad = matrix_from_raw(m6->grad, 10, 10);
	// matrix_print(m6grad);

} 





