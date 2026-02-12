#include "matrix.h"
#include "matrix_math.h"
#include "autograd.h"
#include <immintrin.h>
#include <stdlib.h>
#include <time.h>
#define INT 0.50
#define ROWS INT*1024
#define COLS INT*1024
#define MU 0.0
#define SIGMA 1
#define REQUIRES_GRAD 0


// matrix* add_matrices(matrix* inp1, matrix* inp2){
// 	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad || inp2->requires_grad);
// 	size_t vector_limit = (inp1->cols / 4) * 4;
// 	printf("%li\n", vector_limit);
// 	for(size_t i = 0; i < inp1->rows; i++){
// 		double* d1 = inp1->data + i*inp1->stride;
// 		double* d2 = inp2->data + i*inp2->stride;
// 		double* o = out->data + i*out->stride;
// 		size_t j = 0;
// 		for(; j < vector_limit; j+=4){
// 			__m256d v1 = _mm256_load_pd(&d1[j]);
// 			__m256d v2 = _mm256_load_pd(&d2[j]);
// 			__m256d vout = _mm256_add_pd(v1, v2);
// 			_mm256_store_pd(&o[j], vout);
// 		}
// 	}
// 	return out;
// }


int main(){
	srand(time(NULL));
	// matrix* m1 = matrix_ones(1, COLS, 0);
	// matrix* m2 = matrix_ones(COLS, 1, 0);
	// matrix* m3 = matrix_matmul(m1, m2);
	// printf("Hello World\n");

	matrix* m1 = matrix_random_normal(5, 5, 0, 1, 0);
	matrix* m2 = matrix_random_uniform(5, 5, -100, 100, 0);
	matrix* out = matrix_mul(m1, m2);

	for(size_t i = 0; i < m1->rows; i++){
		double* i1 = m1->data + i*m1->stride;
		double* i2 = m2->data + i*m2->stride;
		double* o = out->data + i*out->stride;
		for(size_t j = 0; j < m1->cols; j++){
			if(o[j] != i1[j]*i2[j]){
				perror("invalid\n");
				exit(1);

			}

		}


	}





	return 0;
}
