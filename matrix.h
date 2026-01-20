//	Core linear algebra lib for neural network applications
#ifndef MATRIX_H
#define MATRIX_H
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>


typedef struct matrix{
	size_t rows, cols;
	size_t size;
	int* ref_count;
	double *data; 
} matrix;


static inline bool MATRIX_NULL(const matrix* m){
	return ((m==NULL)||(m->data==NULL));
}
static inline size_t offset(const matrix* m, int i, int j){
	return (i*m->cols + j);
}
static inline void MATRIX_ERROR(char* msg){
	perror(msg);
	exit(EXIT_FAILURE);
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


void matrix_print_shape(matrix* m);
double matrix_max(const matrix* m);
double matrix_min(const matrix* m);
double matrix_mean(const matrix* m);
double matrix_std(const matrix* m);
double matrix_sum(const matrix* m);


matrix* matrix_alloc(int ROWS, int COLS);
matrix* matrix_ones(int ROWS, int COLS);
matrix* matrix_linspace(double start, double end, size_t num);
matrix* matrix_arange(double start, double end, double step);
void matrix_print(matrix* m);
void matrix_free(matrix* m);
matrix* matrix_transpose(matrix* m);
matrix* matrix_copy(const matrix* input);
matrix* matrix_reshape(matrix* m, size_t ROWS, size_t COLS);


//	Elementwise addition for two matrix_scale
void matrix_add(matrix* a, matrix* b, matrix* c);
void matrix_scale(matrix* a, double b);
void matrix_sub(matrix* a, matrix* b, matrix* c);
void matrix_hadamard(matrix* a, matrix* b, matrix* c);
bool matrix_equality(matrix* a, matrix* b);
void matrix_map(matrix* m, double (*function)(double x));
void matmul(matrix* a, matrix* b, matrix* c);

void matrix_arithmetic(matrix* inp1, matrix* inp2, matrix* out, double (*function)(double x, double y));
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
matrix* matrix_from_arrays(double** arrays);







#endif // !MATRIX_MATRIX_H
