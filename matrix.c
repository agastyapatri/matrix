#include "matrix.h"
#include <math.h>
#include <stdlib.h> 
#include <stdio.h>
#include <time.h>
#include "functional.h"



matrix* matrix_alloc(int ROWS, int COLS){
	matrix* m = calloc(1, sizeof(matrix));
	m->cols = COLS; 
	m->rows = ROWS; 
	m->size = ROWS*COLS;
	m->ref_count = (int*)calloc(1, sizeof(int));
	(*(m->ref_count))++;
	m->data = (double*)calloc(m->size, sizeof(double));
	return m;
}
matrix* matrix_ones(int ROWS, int COLS){
	matrix* m = matrix_alloc(ROWS,  COLS);
	for(size_t i = 0; i < m->size; i++){
		m->data[i] = 1;
	}
	return m;

}
void matrix_print_shape(matrix* m){
	printf("(%li, %li)\n", m->rows, m->cols);
}

void matrix_free(matrix* m){
	if(MATRIX_NULL(m)){
		MATRIX_ERROR("ERROR: argument in matrix_free() is NULL\n");
	}
	if(*(m->ref_count) == 0){
		MATRIX_ERROR("ERROR: argument in matrix_free() is already freed.\n");
	}
	(*(m->ref_count))--;
	if(*(m->ref_count) == 0){
		free(m->ref_count);
		free(m->data);
	}
	free(m);
}

void matrix_print(matrix *m){
	if(MATRIX_NULL(m)){
		MATRIX_ERROR("ERROR: argument in print_matrix() is NULL\n");
	}
	printf("[");
	for (size_t i = 0; i < m->rows ; i++) {
		for (size_t j = 0; j < m->cols ; j++) {
			printf("%f", m->data[i*m->cols + j]);
			if(!(j == m->cols-1)) printf(", ");
		}
		if(!(i == m->rows-1)) printf("\n");
	}
	printf("]\n");
}


void matmul(matrix* a, matrix* b, matrix* c){
	if(MATRIX_NULL(a) || MATRIX_NULL(b) || !c){
		MATRIX_ERROR("ERROR: matrix argument in matmul() is NULL\n");
	}
	if(a->cols != b->rows){
		MATRIX_ERROR("ERROR: invalid input matrix dimensions in matmul()\n");
	}
	if ((c->rows != a->rows) || (c->cols != b->cols)){
		MATRIX_ERROR("ERROR: invalid output matrix dimensions in matmul()\n");
	} 
	double _dot;
	for(size_t i = 0; i < a->rows; i++){
		for(size_t j = 0; j < b->cols; j++){
			_dot = 0;
			for(size_t k = 0; k < a->cols; k++){
				_dot += a->data[i*a->cols + k]*b->data[k*b->cols + j];
			}
			set(c, _dot, i, j);
		} 
	}
}

matrix* matrix_transpose(matrix* m){
	if(MATRIX_NULL(m))
		MATRIX_ERROR("NULL argument passed to matrix_transpose()\n");
	matrix* out = (matrix*)malloc(sizeof(matrix));
	out->rows = m->cols;
	out->cols = m->rows;
	out->size = m->size;
	out->data = m->data;
	out->ref_count = m->ref_count;
	(*(m->ref_count))++;
	return out;
}

matrix* matrix_reshape(matrix* m, size_t ROWS, size_t COLS){
	if(MATRIX_NULL(m))
		MATRIX_ERROR("NULL argument passed to matrix_reshape()\n");
	if(m->rows*m->cols != ROWS*COLS)
		MATRIX_ERROR("Invalid ROWS or COLS argument(s) passed to matrix_reshape(); Ensure that the cardinality of the intended matrix is the same as the target matrix\n");
	matrix* out = (matrix*)malloc(sizeof(matrix));
	out->rows = ROWS;
	out->cols = COLS;
	out->size = m->size;
	out->data = m->data;
	out->ref_count = m->ref_count;
	(*(m->ref_count))++;
	return out;
}


void matrix_add(matrix* a, matrix* b, matrix* c){
	if(MATRIX_NULL(a) || MATRIX_NULL(b) || !c){
		MATRIX_ERROR("ERROR: matrix argument(s) in add_matrix() is NULL\n");
	}
	if(a->cols != b->cols || a->rows != b->rows){
		MATRIX_ERROR("ERROR: invalid input matrix dimensions in add_matrix()\n");
	}
	if ((c->rows != a->rows)  || (c->cols != a->cols)){
		MATRIX_ERROR("ERROR: invalid output matrix dimensions in add_matrix()\n");
	} 
	for(size_t i = 0; i < a->rows; i++){
		for(size_t j = 0; j < b->cols; j++){
			c->data[i*c->cols + j] = a->data[i*a->cols + j] + b->data[i*a->cols + j] ;
			// set(c, (get(a, i, j) + get(b, i, j)), i, j);
		} 
	}
}

void matrix_scale(matrix* a, double b){
	if(MATRIX_NULL(a)){
		MATRIX_ERROR("ERROR: matrix argument(s) in matrix_scale() is NULL\n");
	}
	for(size_t i = 0; i < a->size; i++){
		a->data[i] = b*a->data[i];
	}
} 

void matrix_sub(matrix* a, matrix* b, matrix* c){
	if(MATRIX_NULL(a) || MATRIX_NULL(b) || !c){
		MATRIX_ERROR("ERROR: matrix argument(s) in sub_matrix() is NULL\n");
	}
	if(a->cols != b->cols || a->rows != b->rows){
		MATRIX_ERROR("ERROR: invalid input matrix dimensions in sub_matrix()\n");
	}
	if ((c->rows != a->rows)  || (c->cols != a->cols)){
		MATRIX_ERROR("ERROR: invalid output matrix dimensions in sub_matrix()\n");
	} 
	for(size_t i = 0; i < a->rows; i++){
		for(size_t j = 0; j < b->cols; j++){
			c->data[i*c->cols + j] = a->data[i*a->cols + j] - b->data[i*a->cols + j] ;
			// set(c, (get(a, i, j) + get(b, i, j)), i, j);
		} 
	}
}
void matrix_hadamard(matrix* a, matrix* b, matrix* c){
	if(MATRIX_NULL(a) || MATRIX_NULL(b) || !c){
		MATRIX_ERROR("ERROR: matrix argument(s) in mul_elemwise_matrix() is NULL\n");
	}
	if(a->cols != b->cols || a->rows != b->rows){
		MATRIX_ERROR("ERROR: invalid input matrix dimensions in mul_elemwise_matrix()\n");
	}
	if ((c->rows != a->rows)  || (c->cols != a->cols)){
		MATRIX_ERROR("ERROR: invalid output matrix dimensions in mul_elemwise_matrix()\n");
	} 
	for(size_t i = 0; i < a->rows; i++){
		for(size_t j = 0; j < b->cols; j++){
			c->data[i*c->cols + j] = a->data[i*a->cols + j] * b->data[i*a->cols + j] ;
		} 
	}
}

bool matrix_equality(matrix* a, matrix* b){
	if(MATRIX_NULL(a) || MATRIX_NULL(b)){
		MATRIX_ERROR("ERROR: matrix argument(s) in matrix_equality() is NULL\n");
	}
	if(a->cols != b->cols || a->rows != b->rows){
		return false;
	}
	for(size_t i = 0; i < a->rows; i++){
		for(size_t j = 0; j < a->cols; j++){
			if(!(a->data[i*a->cols + j] == b->data[i*b->cols + j])){
				return false;
			}
		} 
	}
	return true;

}


void matrix_randomize(matrix* m, double (*function)(double mu, double sigma)){
	if(MATRIX_NULL(m)){
		MATRIX_ERROR("NULL argument passed to matrix_randomize()\n");
	}
	for(size_t i = 0; i < m->size; i++){
		m->data[i] = function(0, 1);
	}
}

matrix* matrix_random_uniform(int ROWS, int COLS, double left, double right){
	matrix* m = matrix_alloc( ROWS,  COLS);
	for(size_t i = 0; i < m->size; i++){
		m->data[i] = matrix_rand_uniform(left, right);
	}
	return m;
}


// double square(double x){
// 	return x*x;
// }


matrix* matrix_random_normal(int ROWS, int COLS, double mu, double sigma){
	matrix* m = matrix_alloc(ROWS, COLS);
	for(size_t i = 0; i < m->size; i++)
		m->data[i] = matrix_rand_normal(mu, sigma);
	return m;
}


void matrix_add_rowwise(matrix* mat,  matrix* vec, matrix* out){
	if(MATRIX_NULL(mat) || MATRIX_NULL(vec)){
		MATRIX_ERROR("Invalid / NULL arguments to matrix_add_rowwise()\n");
	}
	if(vec->cols != mat->cols){
		MATRIX_ERROR("Invalid matrix dimensions in matrix_add_rowwise(); number of columns are not equal\n");
	}
	for (size_t i = 0; i < out->rows; i++) {
		for (size_t j = 0; j < out->cols; j++) {
			out->data[i*out->cols + j] = vec->data[j]+mat->data[i*mat->cols + j]; 
		} 
	}
}
void matrix_add_colwise(matrix* mat, matrix* vec, matrix* out){
	if(MATRIX_NULL(mat) || MATRIX_NULL(vec)){
		MATRIX_ERROR("Invalid / NULL arguments to matrix_add_colwise()\n");
	}
	if(vec->rows != mat->rows){
		MATRIX_ERROR("Invalid matrix dimensions in matrix_add_colwise(); number of rows are not equal\n");
	}
	for (size_t i = 0; i < out->rows; i++) {
		for (size_t j = 0; j < out->cols; j++) {
			out->data[i*out->cols + j] = vec->data[i]+mat->data[i*mat->cols + j]; 
		} 
	}

}

void matrix_scalar_mul(matrix* input, double scalar, matrix* output){
	if(MATRIX_NULL(input) || MATRIX_NULL(output)){
		MATRIX_ERROR("Invalid matrix argument(s) in matrix_scalar_mul()\n");
	}
	if((input->rows != output->rows) || (input->cols != output->cols)){
		MATRIX_ERROR("Invalid matrix dimensions provided in the arguments of matrix_scalar_mul()\n");

	}
	for(size_t i = 0; i < input->size; i++){
		output->data[i] = scalar*input->data[i];
	}
}

matrix* matrix_copy(const matrix* input){
	if(MATRIX_NULL(input)){
		MATRIX_ERROR("Invalid matrix argument(s) in matrix_copy()\n");
	}
	matrix* output = matrix_alloc(input->rows, input->cols);
	for(size_t i = 0; i < output->size; i++){
		output->data[i] = input->data[i];
	}
	return output;
}


void matrix_map(matrix *m, double (*function)(double)){
	if(MATRIX_NULL(m)){
		MATRIX_ERROR("NULL matrix argument(s) in matrix_map()\n");
	}
	for(size_t i = 0; i < m->size; i++)
		m->data[i] = function(m->data[i]);
} 

void matrix_arithmetic(matrix* inp1, matrix* inp2, matrix* out, double (*function)(double x, double y)){
	if(MATRIX_NULL(inp1) || MATRIX_NULL(inp2) || MATRIX_NULL(out)){
		MATRIX_ERROR("NULL matrix argument(s) in matrix_arithmetic()\n");
	}
	if((inp1->rows != inp2->rows) || (inp1->cols != inp2->cols)){
		MATRIX_ERROR("Invalid input matrix shapes in matrix_arithmetic()\n");
	}
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = function(inp1->data[i], inp2->data[i]);
}



double matrix_max(const matrix* m){
	if(MATRIX_NULL(m) || *(m->ref_count) == 0){
		MATRIX_ERROR("matrix passed to matrix_max() is NULL or already has been freed.\n");
	}
	double max = m->data[0];
	for(size_t i = 1; i < m->size; i++){
		if(max <= m->data[i]) 
			max = m->data[i];
	}
	return max;
}

double matrix_min(const matrix* m){
	if(MATRIX_NULL(m) || *(m->ref_count) == 0){
		MATRIX_ERROR("matrix passed to matrix_max() is NULL or already has been freed.\n");
	}
	double max = m->data[0];
	for(size_t i = 1; i < m->size; i++){
		if(max > m->data[i]) 
			max = m->data[i];
	}
	return max;
}

double matrix_mean(const matrix* m){
	if(MATRIX_NULL(m) || *(m->ref_count) == 0){
		MATRIX_ERROR("matrix passed to matrix_max() is NULL or already has been freed.\n");
	}
	double mean = 0;
	for(size_t i = 0; i < m->size; i++){
		mean += m->data[i];
	}
	return mean/m->size;

}
double matrix_std(const matrix* m){
	if(MATRIX_NULL(m) || *(m->ref_count) == 0){
		MATRIX_ERROR("matrix passed to matrix_max() is NULL or already has been freed.\n");
	}
	double mu  = 0;
	for(size_t i = 0; i < m->size; i++){
		mu += m->data[i];
	}
	mu /= m->size;
	double sigma = 0;
	for(size_t i = 0; i < m->size; i++){
		sigma = (m->data[i] - mu) * (m->data[i] - mu);
	}
	sigma /= m->size;
	return sqrt(sigma);
}

double matrix_sum(const matrix* m){
	if(MATRIX_NULL(m) || *(m->ref_count) == 0){
		MATRIX_ERROR("matrix passed to matrix_max() is NULL or already has been freed.\n");
	}
	double sum = 0;
	
	for(size_t i = 0; i < m->size; i++)
		sum += m->data[i];
	return sum;
}

matrix* matrix_linspace(double start, double end, size_t num){
	if(end < start){
		MATRIX_ERROR("Invalid argument(s) in matrix_linspace(); Ensure end > start\n");
	}
	matrix* out = matrix_alloc(1, num);
	double step = (end - start)/num;
	for(size_t i = 0; i < num; i++){
		out->data[i] = start + i*step; 
	}
	return out;
}

matrix* matrix_arange(double start, double end, double step){
	if((end < start) || step <= 0){
		MATRIX_ERROR("Invalid argument(s) in matrix_linspace(); Ensure end > start and step >= 0\n");
	}
	size_t num = (end - start)/step;
	matrix* out = matrix_alloc(1, num);
	for(size_t i = 0; i < num; i++){
		out->data[i] = start + i*step; 
	}
	return out;
}
