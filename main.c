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
	matrix* b = matrix_sin(a);
	matrix* c = matrix_relu(b);


	matrix_print(b);
	printf("\n");
	matrix_print(c);

	// matrix_one_grad(c);
	// matrix_grad(c);
	// for(size_t i = 0; i < c->size; i++)
	// 	printf("%lf\t%lf\n", b->grad[i], d->data[i]);

	return 0;
}
