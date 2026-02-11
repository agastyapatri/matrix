#include "matrix.h"
#include "matrix_math.h"
#include "autograd.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define INT 0.50
#define ROWS INT*1024
#define COLS INT*1024
#define MU 0.0
#define SIGMA 1
#define REQUIRES_GRAD 0




int main(){
	srand(time(NULL));
	matrix* a = matrix_arange(-1, 1, 0.1, 1);
	matrix* b = matrix_arange(-1, 1, 0.1, 1);
	matrix* c = matrix_add(a, b);
	matrix* d = matrix_sin(c);
	matrix* f = matrix_cos(c);
	matrix* e = matrix_cos(d);
	matrix_one_grad(e);
	matrix_grad(e);
	for(size_t i = 0; i < e->size; i++)
		printf("%lf\t%lf\n", d->grad[i], f->data[i]);






	// for(size_t i = 0; i < c->size; i++)
	// 	d->grad[i] = 1.0;
	//
	// matrix_grad(d);
	// for(int i = 0; i < 5; i++){
	// 	for(int j = 0; j < 5; j++){
	// 		printf("%lf ", c->grad[i*5 + j]);
	// 	} 
	// }










	
	






	return 0;
}
