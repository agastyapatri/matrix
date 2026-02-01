#include "matrix.h"
#include "matrix_math.h"
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
	matrix* m = matrix_ones(1, COLS, 0);
	matrix* n = matrix_ones(COLS, 1, 0);
	matrix* o = matrix_matmul(m, n);








	
	







	return 0;
}
