#include "matrix.h"
#include "matrix_math.h"
#include "autograd.h"
#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define ROWS 3 
#define COLS 4
#define INT 100
int main(){
	srand(0);
	matrix* m1 = matrix_random_normal(ROWS, COLS, 0, 1, 1);
	matrix* m2 = matrix_random_normal(ROWS, COLS, 0, 1, 1);
	matrix* m3 = matrix_sin(m1);
	matrix* m4 = matrix_cos(m2);
	matrix* m5 = matrix_add(m3, m4);
	matrix_backward(m5);


} 





