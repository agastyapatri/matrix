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



void matmul(double* inp1, double* inp2, double* out, size_t inp1rows, size_t inp1cols, size_t inp1stride, size_t inp2cols, size_t inp2stride, size_t outstride){
	for(size_t i = 0; i < inp1rows; i++){
		double* inp1row = inp1 + (i * inp1stride);
		double* outrow = out + (i * outstride);
		for(size_t j = 0; j < inp2cols; j++){
			for(size_t k = 0; k < inp1cols; k++){
				outrow[j] += inp1row[k] * (*(inp2 + k*inp2stride + j));
			}
		}
	}
}

int main(){
	srand(0);
	matrix* inp1 = matrix_random_normal(5, 5, 0, 1, 1);
	matrix* inp2 = matrix_ones(5, 5, 1);
	matrix* inp3 = matrix_matmul(inp1, inp2);
	matrix* out = matrix_alloc(inp1->rows, inp2->cols, 0);

	matmul(inp1->data, inp2->data, out->data, inp1->rows, inp1->cols, inp1->stride, inp2->cols, inp2->stride, out->stride);
	matrix_print(out);





	

	//	confirming if the gradient of the root node with respect to any arbitrary node is as expected
	// matrix_grad(inp2);
	// for(size_t i = 0; i < inp1->rows; i++){
	// 	double* row = inp1->data + (i * inp1->stride);
	// 	for(size_t j = 0; j < inp1->cols; j++){
	// 		printf("%lf ", dsigmoid(row[j]));
	// 	} 
	// 	printf("\n");
	// }
	// printf("\n");
	// for(size_t i = 0; i < inp1->rows; i++){
	//
	// 	double* row = inp1->grad + (i * inp1->stride);
	// 	for(size_t j = 0; j < inp1->cols; j++){
	// 		printf("%lf ", row[j]);
	// 	} 
	// 	printf("\n");
	// }
} 
