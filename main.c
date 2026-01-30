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

matrix* matrix_add_alternate(matrix* inp1, matrix* inp2){
	matrix* out = matrix_alloc(inp1->rows, inp2->cols);
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = inp1->data[i] + inp2->data[i];

	if(inp1->requires_grad || inp2->requires_grad){
		matrix_grad_on(out);
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		out->op = ADD;
		(*(inp1->ref_count))++;
		(*(inp2->ref_count))++;
	}
	return out;
}
matrix* matrix_sub_alternate(matrix* inp1, matrix* inp2){
	matrix* out = matrix_alloc(inp1->rows, inp2->cols);
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = inp1->data[i] - inp2->data[i];

	if(inp1->requires_grad || inp2->requires_grad){
		matrix_grad_on(out);
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		out->op = SUB;
		(*(inp1->ref_count))++;
		(*(inp2->ref_count))++;
	}
	return out;
}
matrix* matrix_elemmul_alternate(matrix* inp1, matrix* inp2){
	matrix* out = matrix_alloc(inp1->rows, inp2->cols);
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = inp1->data[i] * inp2->data[i];

	if(inp1->requires_grad || inp2->requires_grad){
		matrix_grad_on(out);
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		out->op = MUL;
		(*(inp1->ref_count))++;
		(*(inp2->ref_count))++;
	}
	return out;
}

matrix* matmul_alternate(matrix* inp1, matrix* inp2){
	matrix* out = matrix_alloc(inp1->rows, inp2->cols);
	for(size_t bi = 0; bi < inp1->rows; bi+=BLOCK_SIZE){
		for(size_t bk = 0; bk < inp1->cols; bk+=BLOCK_SIZE){
			for(size_t bj = 0; bj < inp2->cols; bj+=BLOCK_SIZE){

				for(size_t i = bi; (i < inp1->rows) && (i < bi + BLOCK_SIZE); i++){
					for(size_t k = bk; (k < inp1->cols) && (k < bk + BLOCK_SIZE); k++){
						double r = inp1->data[offset(inp1, i, k)];
						for(size_t j = bj; (j < inp2->cols) && (j < bj + BLOCK_SIZE); j++){
							out->data[offset(out, i, j)] += r*inp2->data[offset(inp2, k, j)];
						}
					} 
				}
			} 
		} 
	}
	out->requires_grad = inp1->requires_grad || inp2->requires_grad;
	if(out->requires_grad){
		out->op = MATMUL;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		(*(inp1->ref_count))++;
		(*(inp2->ref_count))++;
	}
	return out;
}


int main(){
	matrix* m = matrix_alloc(5, 5);
	matrix_print(m);
	return 0;
}
