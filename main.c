#include "matrix.h"
#include <time.h>
#define ROWS 5
#define COLS 5
#define MU 0.0
#define SIGMA 1


int main(){
	srand(time(NULL));
	matrix* mat = matrix_random_normal(ROWS, COLS, 0., 1.);
	matrix* mat2 = matrix_alloc(ROWS, COLS);
	matrix* mat3 = matrix_alloc(ROWS, COLS);
	matrix* mat4 = matrix_alloc(ROWS, COLS);
	matrix* mat5 = matrix_alloc(ROWS, COLS);
	matrix_arithmetic(mat, mat2, mat3, ADD);
	matrix_arithmetic(mat2, mat3, mat4, MUL);
	matrix_arithmetic(mat4, mat3, mat5, DIV);

	matrix_print(mat5->previous[0]->previous[0]);
	matrix_print(mat5->previous[0]->previous[1]);


	return 0;
}
