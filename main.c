#include "matrix.h"
#include "matrix_math.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define INT 0.25
#define ROWS INT*1024
#define COLS INT*1024
#define MU 0.0
#define SIGMA 1
#define REQUIRES_GRAD 0


static inline void MATRIX_ADD_SIMD(matrix* inp1, matrix* inp2, matrix* out){
	for(size_t i = 0; i < inp1->rows; i++){
		double* d1 = inp1->data + (i * inp1->stride);
		double* d2 = inp2->data + (i * inp2->stride);
		double* o = out->data + (i * out->stride);
		size_t j = 0;
		for(; j <= inp1->cols - 4; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_add_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < inp1->cols; j++){
			o[j] = d1[j] + d2[j];
		}
	}
}



int main(){
	matrix* m = matrix_random_uniform(ROWS, COLS, 0, 1, REQUIRES_GRAD);
	matrix* n = matrix_random_uniform(COLS, ROWS, 0, 1, REQUIRES_GRAD);
	matrix* o = matrix_alloc(COLS, ROWS, IN_COMP_GRAPH(m, n));
	// MATRIX_TIMER(MATRIX_ADD(m, n, o))
	MATRIX_TIMER(MATRIX_ADD_SIMD(m, n, o))







	return 0;
}
