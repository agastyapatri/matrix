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

matrix* matrix_map(matrix* inp, double (*func)(double)){
	matrix* out = matrix_alloc(inp->rows, inp->cols, inp->requires_grad);
	for(size_t i = 0; i < inp->rows; i++){
		double* p1datarow = inp->data + (i * inp->stride);
		double* o1datarow = out->data + (i * inp->stride);
		for(size_t j = 0; j < inp->cols; j++){
			o1datarow[j] = func(p1datarow[j]);
		}
	}
	return out;
}





int main(){
	srand(0);
	matrix* m1 = matrix_random_normal(1, 10, 0, 1, 0);
	matrix* m2 = matrix_random_normal(1, 10, 0, 1, 0);
	matrix* out = matrix_transpose(m2);
	matrix_print(m2);
	matrix_print(out);
} 
