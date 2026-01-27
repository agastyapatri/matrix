#include "matrix.h"
#include "matrix_math.h"
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#define INT 1
#define ROWS INT*1024 
#define COLS INT*1024 
#define MU 0.0
#define SIGMA 1


// void simd_matmul(matrix* inp1, matrix* inp2, matrix* out){
// 	size_t M = inp1->rows;
// 	size_t K = inp1->cols;
// 	size_t N = inp2->cols;
// 	for(size_t bi = 0; bi < M; bi+=BLOCK_SIZE){
// 		for(size_t bk = 0; bk < K; bk+=BLOCK_SIZE){
// 			for(size_t bj = 0; bj < N; bj+=BLOCK_SIZE){
//
// 				for(size_t i = bi; (i < M) && (i < bi + BLOCK_SIZE); i++){
// 					for(size_t k = bk; (k < K) && (k < bk + BLOCK_SIZE); k++){
// 						__m256 r_vec = _mm256_set1_pd(inp1->data[offset(inp1, i, k)]);
// 						for(size_t j = bj; (j < N) && (j < bj + BLOCK_SIZE); j++){
// 							__m256d c_vec = _mm256_loadu_pd(&out->data[offset(out, i, j)]);
// 							__m256d b_vec = _mm256_loadu_pd(&inp2->data[offset(out, k, j)]);
// 							c_vec = _mm256_fmadd_pd(r_vec, b_vec, c_vec);
// 							_mm256_storeu_pd(&out->data[offset(out, i, j)], c_vec);
// 						}
// 					} 
// 				}
// 			} 
// 		} 
// 	}
// }



int main(){
	matrix* m = matrix_alloc(16, 784);
	matrix* n = matrix_alloc(784, 392);
	matrix* o = matrix_alloc(392, 196);
	matrix* p = matrix_alloc(196, 49);
	matrix* q = matrix_alloc(49, 10);


	TIMER(matmul(m, n, o));


	matrix_free(m);
	matrix_free(n);
	matrix_free(o);
	matrix_free(p);
	matrix_free(q);



}
