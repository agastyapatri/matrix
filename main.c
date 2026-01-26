#include "matrix.h"
#include "matrix_math.h"
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#define ROWS 1024 
#define COLS 1024 
#define MU 0.0
#define SIGMA 1





void simd_matmul(matrix* inp1, matrix* inp2, matrix* out){
	size_t M = inp1->rows;
	size_t K = inp1->cols;
	size_t N = inp2->cols;
	for(size_t bi = 0; bi < M; bi+=BLOCK_SIZE){
		for(size_t bk = 0; bk < K; bk+=BLOCK_SIZE){
			for(size_t bj = 0; bj < N; bj+=BLOCK_SIZE){

				for(size_t i = bi; (i < M) && (i < bi + BLOCK_SIZE); i++){
					for(size_t k = bk; (k < K) && (k < bk + BLOCK_SIZE); k++){
						__m256 r_vec = _mm256_set1_pd(inp1->data[offset(inp1, i, k)]);
						for(size_t j = bj; (j < N) && (j < bj + BLOCK_SIZE); j++){
							__m256d c_vec = _mm256_loadu_pd(&out->data[offset(out, i, j)]);
							__m256d b_vec = _mm256_loadu_pd(&inp2->data[offset(out, k, j)]);
							c_vec = _mm256_fmadd_pd(r_vec, b_vec, c_vec);
							_mm256_storeu_pd(&out->data[offset(out, i, j)], c_vec);
						}
					} 
				}
			} 
		} 
	}
}



void add_vec_simd(double* a, double* b, double* res, size_t len, int block_size){
	for(size_t i = 0; i <= len-block_size; i+=block_size){
		__m256d va = _mm256_loadu_pd(&a[i]);
		__m256d vb = _mm256_loadu_pd(&b[i]);
		__m256d vr = _mm256_add_pd(va, vb);
		_mm256_storeu_pd(&res[i], vr);
	}
}


void add_vec(double* a, double* b, double* res, size_t len){
	for(size_t i = 0; i < len; i++){
		res[i] = a[i] + b[i];
	}
}


int main(){
	// srand(time(NULL));
	srand(0);
	// matrix* a = matrix_random_uniform(1, 16, 0, 1);
	// matrix* b = matrix_random_uniform(1, 16, 0, 1);
	// matrix* a = matrix_ones(1, len);
	// matrix* b = matrix_ones(1, len);
	// matrix* res = matrix_alloc(1, len);
	//
	//
	int alignment = 32;
	int block_size = 64;
	int len = 10*1024;
	double* a = (double*)aligned_alloc(alignment, len);
	double* b = (double*)aligned_alloc(alignment, len);
	double* res = (double*)aligned_alloc(alignment, len);
	for(int i = 0; i < len; i++){
		a[i] = 1;
		b[i] = 1;
		res[1] = 0;
	}







	

	// TIMER(add_vec(a, b, res, len));
	TIMER(add_vec_simd(a, b, res, len, block_size));




	// TIMER(add_vec(a, b , res, len));




	// for(; i < len; i++){
	// 	res->data[i] = a->data[i] + b->data[i];
	// } 

	return 0;
}
