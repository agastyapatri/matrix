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



int main(){
	srand(time(NULL));
	/*
	 *	a = 10 
	 *	b = 20 ; b->grad = dd/dc * dc/db = c->grad * 1 
	 *	c = a + b; c->grad = dd/dc = d->grad * b 
	 *	d = c * b; d->grad dd/ dd = 1 
	 *
	 */ 

	matrix* inp1 = matrix_ones(5, 5, 1);	//	a  
	matrix* inp2 = matrix_random_normal(5, 5, -1, 2, 1);	//b  
	matrix* inp3 = matrix_add(inp1, inp2);	// c = a + b 
	matrix* inp4 = matrix_mul(inp3, inp2);	// d = c*b 
    matrix_grad(inp4);
	// printf("%d\n", inp4->num_prevs);


	// for(size_t i = 0; i < inp3->rows; i++){
	// 	double* row = inp3->data + (i * inp3->stride);
	// 	for(size_t j = 0; j < inp3->cols; j++){
	// 		printf("%lf ", row[j]);
	// 	} 
	// 	printf("\n");
	//
	// }
















	return 0;
}
