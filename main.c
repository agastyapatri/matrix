#include "matrix.h"
#include "matrix_math.h"
#include "autograd.h"
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
	matrix* inp1 = matrix_linspace(-5, 5, 100, 1);
	matrix* inp2 = matrix_sigmoid(inp1);
	matrix* inp3 = matrix_sin(inp2);
	matrix_grad(inp3);
	

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
