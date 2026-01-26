#include "matrix.h"
#include <math.h>
#include <stdlib.h> 
#include <stdio.h>
#include <time.h>
#include "matrix_math.h"




matrix* matrix_alloc(int ROWS, int COLS){
	matrix* m = calloc(1, sizeof(matrix));
	m->cols = COLS; 
	m->rows = ROWS; 
	m->size = ROWS*COLS;
	m->ref_count = (int*)calloc(1, sizeof(int));
	(*(m->ref_count))++;
	m->data = (double*)calloc(m->size, sizeof(double));
	// m->data = (double*)aligned_alloc()

	m->requires_grad = 0; 
	m->grad = NULL;
	m->op = NONE;
	m->num_prevs = 0;
	return m;
}


matrix* matrix_aligned_alloc(int ROWS, int COLS, bool requires_grad){
	matrix* m = malloc(sizeof(matrix));
	if(MATRIX_NULL(m)) 
		return NULL; 
	m->rows = ROWS;
	m->cols = COLS;
	m->size = ROWS*COLS;
	m->requires_grad = requires_grad;
	m->ref_count = (int*)malloc(sizeof(size_t));
	*(m->ref_count) = 1;
	size_t bytes = sizeof(double)*m->size;
	bytes += ALIGNMENT - (bytes % ALIGNMENT);
	m->data = (double*)aligned_alloc(ALIGNMENT, bytes);
	if(m->requires_grad){
		m->grad =(double*)aligned_alloc(ALIGNMENT, bytes); 
		for(size_t i = 0; i < m->size; i++){
			m->grad[i] = 0.0;
		}
	}
	else{
		m->grad = NULL;
	}
	return m;
}



void matrix_grad_on(matrix* m){
	m->requires_grad = 1; 
	m->grad = (double*)calloc(m->size, sizeof(double)); 
}

void matrix_grad_off(matrix* m){
	m->requires_grad = 0; 
	m->op = NONE;
	free(m->grad);
	m->grad = NULL;
}



matrix* matrix_ones(int ROWS, int COLS){
	matrix* m = matrix_alloc(ROWS,  COLS);
	for(size_t i = 0; i < m->size; i++){
		m->data[i] = 1;
	}
	return m;
}

matrix* matrix_eye(int SIDE){
	matrix* m = matrix_alloc(SIDE, SIDE);
	for(size_t i = 0; i < m->rows; i++){
		for(size_t j = 0; j < m->cols; j++){
			m->data[offset(m, i, j)] = (i == j ) ? 1 : 0;
		} 
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
	char* opstring = get_optype_string(m->op);
	printf("matrix(\n\t[");
	for (size_t i = 0; i < m->rows ; i++) {
		if(i>=1) printf("\t");
		printf("[");
		for (size_t j = 0; j < m->cols ; j++) {
			printf("%f", m->data[i*m->cols + j]);
			if(!(j == m->cols-1)) printf(", ");
		}
		printf("]");
		if(!(i == m->rows-1)) printf(",\n");
	};
	printf("],\nrequires_grad = %d, optype = %s)\n", m->requires_grad, opstring);
}


void matmul(matrix* inp1, matrix* inp2, matrix* out){
	for(size_t bi = 0; bi < inp1->rows; bi+=BLOCK_SIZE){
		for(size_t bk = 0; bk < inp1->cols; bk+=BLOCK_SIZE){
			for(size_t bj = 0; bj < inp2->cols; bj+=BLOCK_SIZE){

				for(size_t i = bi; (i < inp1->rows) && (i < bi + BLOCK_SIZE); i++){
					for(size_t k = bk; (k < inp1->cols) && (k < bk + BLOCK_SIZE); k++){
						double r = inp1->data[offset(inp1, i, k)];
						for(size_t j = bj; (j < inp2->cols) && (j < bj + BLOCK_SIZE); j++){
							out->data[offset(out, i, j)] += r*inp2->data[offset(inp2, k, j)];
						}
					} 
				}


			} 
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

void matrix_scale(matrix* a, double b){
	for(size_t i = 0; i < a->size; i++){
		a->data[i] = b*a->data[i];
	}
} 



void matrix_hadamard(matrix* a, matrix* b, matrix* c){
	for(size_t i = 0; i < a->rows; i++){
		for(size_t j = 0; j < b->cols; j++){
			c->data[i*c->cols + j] = a->data[i*a->cols + j] * b->data[i*a->cols + j] ;
		} 
	}
}

bool matrix_equality(matrix* a, matrix* b){
	if(a->cols != b->cols || a->rows != b->rows){
		return false;
	}
	for(size_t i = 0; i < a->rows; i++){
		for(size_t j = 0; j < a->cols; j++){
			if(a->data[i*a->cols + j] != b->data[i*b->cols + j]){
				return false;
			}
		} 
	}
	return true;

}

bool matrix_shape_equality(matrix* a, matrix* b){
	if(a->cols != b->cols || a->rows != b->rows){
		return false;
	}
	return true;
}

void matrix_randomize(matrix* m, double (*function)(double mu, double sigma)){
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


void matrix_map(matrix* inp1, matrix* out, OPTYPE operation){
	if(MATRIX_NULL(inp1) || MATRIX_NULL(out)){
		MATRIX_ERROR("NULL matrix argument(s) in matrix_map()\n");
	}
	if((inp1->rows != out->rows) || (inp1->cols != out->cols)){
		MATRIX_ERROR("Invalid input matrix shapes in matrix_map()\n");
	}
	unary_op function  = get_unary_operation(operation);
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = function(inp1->data[i]);
	matrix_grad_on(out);
	out->op = operation;
	out->num_prevs = 1;
	out->previous[0] = inp1; 
} 

void matrix_arithmetic(matrix* inp1, matrix* inp2, matrix* out, OPTYPE operation){
	if(MATRIX_NULL(inp1) || MATRIX_NULL(inp2) || MATRIX_NULL(out)){
		MATRIX_ERROR("NULL matrix argument(s) in matrix_arithmetic()\n");
	}
	if((inp1->rows != inp2->rows) || (inp1->cols != inp2->cols)){
		MATRIX_ERROR("Invalid input matrix shapes in matrix_arithmetic()\n");
	}
	binary_op function = get_binary_operation(operation);
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = function(inp1->data[i], inp2->data[i]);
	matrix_grad_on(out);
	out->op = operation;
	out->num_prevs = 2;
	out->previous[0] = inp1; 
	out->previous[1] = inp2;
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

double matrix_det(const matrix* m){
	if(MATRIX_NULL(m))
		MATRIX_ERROR("NULL matrix in matrix_determinant()\n");
	if(m->rows == 2 && m->cols == 2){
		return (get(m, 0, 0)*get(m, 1, 1))-(get(m, 0, 1)*get(m, 1, 0));
	}
	return 0;
}

matrix* matrix_inverse(const matrix* m){
	matrix* out = matrix_alloc(m->rows, m->cols);
	if(m->rows == 2 && m->cols == 2){
		set(out, (1/matrix_det(m))*(get(m, 1, 1)), 0, 0) ; 
		set(out, (1/matrix_det(m))*(-get(m, 0, 1)), 0, 1) ; 
		set(out, (1/matrix_det(m))*(-get(m, 1, 0)), 1, 0) ; 
		set(out, (1/matrix_det(m))*(get(m, 0, 0)), 1, 1) ; 
	}
	return out;
}

double matrix_trace(const matrix* m){
	if(!is_square(m))
		MATRIX_ERROR("Matrix argument is not square in matrix_trace()\n");
	double trace = 0;
	for(size_t i = 0; i < m->rows; i++){
		trace += get(m, i, i);
	}
	return trace;
}

void matrix_push_back(matrix* mat, double* array){
	mat->rows++;
	mat->data = realloc(mat->data, mat->rows*mat->cols*sizeof(double));
	for(size_t i = mat->size; i < mat->rows* mat->cols; i++){
		mat->data[i] = array[i - mat->size];
	}
	mat->size = mat->rows*mat->cols;
}

matrix* matrix_from_arrays(double** arrays, int num_rows, int num_cols);

