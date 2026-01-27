#ifndef MATRIX_H
#define MATRIX_H
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_math.h"

#define MAX_PREVS 3 
#define MAX_ARGS 5 
#define MAX_PARAM_MATRICES 10

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
#ifndef ALIGNMENT
#define ALIGNMENT 64
#endif

#define TIMER(function) clock_t start = clock();\
						function;				\
						clock_t end = clock();  \
						printf("%0.10f\n", (double)(end - start)/CLOCKS_PER_SEC);

#define MATRIX_NULL(m) (m==NULL) ? 1 : 0
#define MATRIX_ERROR(msg) printf(msg);	\
						  exit(EXIT_FAILURE);

typedef struct matrix{
	//	core metadata
	size_t rows, cols;
	size_t size;
	int* ref_count;
	double* data; 
	size_t bytes;
	int stride;
	int padding;
	


	//	autograd metadata
	bool requires_grad;
	double* grad;
	OPTYPE op;
	struct matrix* previous[MAX_PREVS];
	int num_prevs;

} matrix;


static inline size_t offset(const matrix* m, int i, int j){
	return (i*m->cols + j);
}
static inline double get(const matrix* m, int i, int j){
	return m->data[offset(m, i, j)];
}
static inline void set(matrix* m, double val, int i, int j){
	m->data[i*m->cols + j] = val;
}
static inline bool is_square(const matrix* m){
	return (m->rows == m->cols) ? 1 : 0;
}
static inline bool matrix_shape_equality(matrix* a, matrix* b){
	if(a->cols != b->cols || a->rows != b->rows){
		return false;
	}
	return true;
}


void matrix_print_shape(matrix* m);
matrix* matrix_max(const matrix* m);
matrix* matrix_min(const matrix* m);
matrix* matrix_mean(const matrix* m);
matrix* matrix_std(const matrix* m);
matrix* matrix_sum(const matrix* m);
void matrix_grad_on(matrix* m);
void matrix_grad_off(matrix* m);


matrix* matrix_alloc(int ROWS, int COLS);
// matrix* matrix_aligned_alloc(int ROWS, int COLS);
matrix* matrix_ones(int ROWS, int COLS);
matrix* matrix_eye(int SIDE);
matrix* matrix_linspace(double start, double end, size_t num);
matrix* matrix_arange(double start, double end, double step);
void matrix_print(matrix* m);
void matrix_free(matrix* m);
matrix* matrix_transpose(matrix* m);
matrix* matrix_copy(const matrix* input);
matrix* matrix_reshape(matrix* m, size_t ROWS, size_t COLS);


void matrix_scale(matrix* a, double b);
void matrix_hadamard(matrix* a, matrix* b, matrix* c);
bool matrix_equality(matrix* a, matrix* b);
void matmul(matrix* inp1, matrix* inp2, matrix* out);

void matrix_unary_op(matrix* inp1, matrix* out, OPTYPE operation);
void matrix_binary_op(matrix* inp1, matrix* inp2, matrix* out, OPTYPE operation);


//	Replaces the elements of an existing matrix with random elements between -1 and 1
void matrix_randomize(matrix* m, double (*function)(double, double));

//	Creates a matrix with random elements between -1 and 1
matrix* matrix_random_uniform(int ROWS, int COLS, double left, double right);
matrix* matrix_random_normal(int ROWS, int COLS, double mu, double sigma);


//	Adds a matrix and a row vector by row-wise broadcasting of the row vector (1xm) matrix
void matrix_add_rowwise(matrix* mat, matrix* vec, matrix* out);

//	Adds a matrix and a column vector by col-wise broadcasting of the column vector (mx1) matrix
void matrix_add_colwise(matrix* mat, matrix* vec, matrix* out);

matrix* matrix_sort(const matrix* m);
// double matrix_search(const matrix* m, double element);

double matrix_det(const matrix* m);
matrix* matrix_inverse(const matrix* m);
double matrix_trace(const matrix* m);

void matrix_push_back(matrix* mat, double* array);


//TODO 
matrix* matrix_from_arrays(double** arrays, int num_rows, int num_cols);






#endif // !MATRIX_MATRIX_H
