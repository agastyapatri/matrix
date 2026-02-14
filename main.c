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

void matrix_map(matrix* inp, double (*func)(double)){
	for(size_t i = 0; i < inp->rows; i++){
		double* p1gradrow = inp->data + (i * inp->stride);
		for(size_t j = 0; j < inp->cols; j++){
			p1gradrow[j] = func(p1gradrow[j]);
		}
	}
}


int main(){
	srand(time(NULL));

	matrix* inp1 = matrix_ones(11, 11, 1);
	matrix* inp2 = matrix_random_normal(11, 11, -1, 1, 1);
	matrix* inp3 = matrix_mul(inp1, inp2);
	matrix* out = matrix_sin(inp2);
	// matrix* inp5 = matrix_add(inp3, inp4);
	// matrix* out = matrix_exp(inp5);
	matrix_grad(out);






	for(size_t i = 0; i < out->rows; i++){
		double* p1gradrow = out->previous[0]->grad + (i * out->stride);
		for(size_t j = 0; j < out->cols; j++){
			printf("%lf ", p1gradrow[j]);
		}

		printf("\n");
	}
	printf("\n\n");

	matrix_map(inp2, dsin);
	matrix_print(inp2);
	






}
