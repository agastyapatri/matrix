#include "matrix.h"
#include "matrix_math.h"
#include "autograd.h"
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
	// matrix* m1 = matrix_ones(1, COLS, 0);
	// matrix* m2 = matrix_ones(COLS, 1, 0);
	// matrix* m3 = matrix_matmul(m1, m2);
	// printf("Hello World\n");

	matrix* m1 = matrix_ones(5, 5, 0);
	matrix* m2 = matrix_ones(5, 5, 0);

	matrix* m3 = matrix_add(m1, m2);

	// matrix_print(m3);



	return 0;
}
