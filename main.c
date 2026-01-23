#include "matrix.h"
#include "matrix_math.h"
#include <time.h>
#define ROWS 10
#define COLS 10
#define MU 0.0
#define SIGMA 1

void matrelu(matrix* m){
	for(size_t i = 0 ; i < m->size; i++){
		m->data[i] = (m->data[i] > 0) ? m->data[i] : 0;
	}
}


int main(){
	srand(time(NULL));
	matrix* mat = matrix_random_normal(ROWS, COLS, 0., 1.);
	// matrix* mat2 = matrix_alloc(ROWS, COLS);
	// matrix* mat3 = matrix_alloc(ROWS, COLS);
	// matrix* mat4 = matrix_alloc(ROWS, COLS);
	// matrix* mat5 = matrix_alloc(ROWS, COLS);
	TIMER(matrix_arithmetic(mat, mat, mat, DIV));
	matrix_print(mat);







	// matrix_free(mat);
	// matrix_free(mat2);
	// matrix_free(mat3);
	// matrix_free(mat4);
	// matrix_free(mat5);
	return 0;
}
