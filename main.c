#include "matrix.h"
#include "matrix_math.h"
#include <time.h>
#include <immintrin.h>
#define ROWS 2000
#define COLS 2000
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



int main(){
	// srand(time(NULL));
	srand(0);
	matrix* mat = matrix_ones(ROWS, COLS);
	matrix* mat2 = matrix_eye(ROWS);
	matrix* out = matrix_alloc(mat->rows, mat2->cols);
	TIMER(matmul(mat, mat2, out));
	// TIMER(simd_matmul(mat, mat2, out));
	// TIMER(matmul(mat, mat2, out));
	printf("%d\n", matrix_equality(mat, out));




	// TIMER(matrix_arithmetic(mat, mat, mat, MUL));







	// matrix_free(mat);
	// matrix_free(mat2);
	// matrix_free(mat3);
	// matrix_free(mat4);
	// matrix_free(mat5);
	return 0;
}
