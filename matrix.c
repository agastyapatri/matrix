#include "matrix.h"
#include <math.h>
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "matrix_math.h"

matrix* matrix_alloc(int ROWS, int COLS, bool requires_grad){
	matrix* m = malloc(sizeof(matrix));
	if(!m)
		return NULL; 
	m->rows = ROWS;
	m->cols = COLS;
	m->op = NONE;
	m->ref_count = (int*)malloc(sizeof(int));
	m->ref_count[0] = 1;
	m->stride = (m->cols + (ALIGNMENT / sizeof(double)) - 1)  & ~((ALIGNMENT / sizeof(double)) - 1);
	m->padding = m->stride - m->cols;
	m->size = m->rows * m->stride;
	m->bytes = m->size * sizeof(double);
	m->data = (double*)aligned_alloc(ALIGNMENT, m->bytes);
	if(!(m->data))
		return NULL; 
	m->requires_grad = requires_grad;
	if(m->requires_grad){
		m->num_prevs = 0; 
		m->grad = (double*)aligned_alloc(ALIGNMENT, m->bytes);
		if(!(m->grad))
			return NULL; 
	}
	return m;
}


void matrix_grad_on(matrix* m){
	m->requires_grad = true;
	size_t bytes = sizeof(double)*m->size;
	bytes += ALIGNMENT - (bytes % ALIGNMENT);
	m->grad = (double*)aligned_alloc(ALIGNMENT, bytes);
	for(size_t i = 0; i < m->size; i++)
		m->grad[i] = 0.0;
}

void matrix_grad_off(matrix* m){
	m->requires_grad = 0; 
	m->op = NONE;
	if(m->grad){
		free(m->grad);
	}
}


matrix* matrix_ones(int ROWS, int COLS, bool requires_grad){
	matrix* m = matrix_alloc(ROWS,  COLS, requires_grad);
	for(size_t i = 0; i < m->size; i++)
		m->data[i] = 1;
	return m;
}



matrix* matrix_eye(int SIDE, bool requires_grad){
	matrix* m = matrix_alloc(SIDE, SIDE, requires_grad);
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
		if(!m->requires_grad){
			free(m->grad);
		}
	}
	free(m);
}

void matrix_print(matrix *m){
	if(MATRIX_NULL(m)){
		MATRIX_ERROR("ERROR: argument in print_matrix() is NULL\n");
	}
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

	char* opstring = get_optype_string(m->op);
	// char* opstring = "none";
	// printf("],\nrequires_grad = %d, optype = %s)\n", m->requires_grad, opstring);
	printf("]");
	if(m->requires_grad){
		printf("\nrequires_grad = %d, optype = %s", m->requires_grad, opstring);
	}
	printf(")\n");
}



matrix* matrix_matmul(matrix* inp1, matrix* inp2){
	bool reqgrad = inp1->requires_grad || inp2->requires_grad;
	matrix* out = matrix_alloc(inp1->rows, inp2->cols, reqgrad);
	MATRIX_MATMUL(inp1, inp2, out);
	out->requires_grad = inp1->requires_grad || inp2->requires_grad;
	if(out->requires_grad){
		out->op = MATMUL;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		(*(inp1->ref_count))++;
		(*(inp2->ref_count))++;
	}
	return out;
}


matrix* matrix_add(matrix* inp1, matrix* inp2){
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad || inp2->requires_grad);
	MATRIX_ADD(inp1, inp2, out);
	if(out->requires_grad){
		out->op = ADD;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		(*(inp1->ref_count))++;
		(*(inp2->ref_count))++;
	}
	return out;
}

matrix* matrix_pow(matrix* inp1, matrix* inp2){
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad || inp2->requires_grad);
	MATRIX_POW(inp1, inp2, out);
	if(out->requires_grad){
		out->op = POW;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		(*(inp1->ref_count))++;
		(*(inp2->ref_count))++;
	}
	return out;
}

matrix* matrix_sin(matrix* inp1){
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	MATRIX_SIN(inp1, out);
	if(out->requires_grad){
		out->op = SIN;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		(*(inp1->ref_count))++;
	}
	return out;
}

matrix* matrix_tanh(matrix* inp1){
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	MATRIX_TANH(inp1, out);
	if(out->requires_grad){
		out->op = TANH;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		(*(inp1->ref_count))++;
	}
	return out;
}

matrix* matrix_log(matrix* inp1){
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	MATRIX_LOG(inp1, out);
	if(out->requires_grad){
		out->op = LOG;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		(*(inp1->ref_count))++;
	}
	return out;
}
matrix* matrix_exp(matrix* inp1){
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	MATRIX_EXP(inp1, out);
	if(out->requires_grad){
		out->op = EXP;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		(*(inp1->ref_count))++;
	}
	return out;
}

matrix* matrix_cos(matrix* inp1){
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	MATRIX_COS(inp1, out);
	if(out->requires_grad){
		out->op = COS;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		(*(inp1->ref_count))++;
	}
	return out;
}

matrix* matrix_sub(matrix* inp1, matrix* inp2){
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad || inp2->requires_grad);
	MATRIX_SUB(inp1, inp2, out);
	out->requires_grad = inp1->requires_grad || inp2->requires_grad;
	if(out->requires_grad){
		out->op = SUB;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		(*(inp1->ref_count))++;
		(*(inp2->ref_count))++;
	}
	return out;
} 

matrix* matrix_mul(matrix* inp1, matrix* inp2){
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad || inp2->requires_grad);
	MATRIX_MUL(inp1, inp2, out);
	out->requires_grad = inp1->requires_grad || inp2->requires_grad;
	if(out->requires_grad){
		out->op = MUL;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		(*(inp1->ref_count))++;
		(*(inp2->ref_count))++;
	}
	return out;
}

matrix* matrix_div(matrix* inp1, matrix* inp2){
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad || inp2->requires_grad);
	MATRIX_DIV(inp1, inp2, out);
	out->requires_grad = inp1->requires_grad || inp2->requires_grad;
	if(out->requires_grad){
		out->op = DIV;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		(*(inp1->ref_count))++;
		(*(inp2->ref_count))++;
	}
	return out;

}

matrix* matrix_transpose(matrix* m){
	matrix* out = (matrix*)malloc(sizeof(matrix));
	out->rows = m->cols;
	out->cols = m->rows;
	out->size = m->size;
	out->data = m->data;
	out->ref_count = m->ref_count;
	out->requires_grad = m->requires_grad;
	out->grad = m->grad;
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
	out->requires_grad = m->requires_grad;
	out->grad = m->grad;
	(*(m->ref_count))++;
	return out;
}

void matrix_scale(matrix* a, double b){
	for(size_t i = 0; i < a->size; i++){
		a->data[i] = b*a->data[i];
	}
} 


bool matrix_equality(matrix* a, matrix* b){
	if(!(matrix_shape_equality(a, b))){
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

void matrix_randomize(matrix* m, double (*function)(double mu, double sigma)){
	for(size_t i = 0; i < m->size; i++){
		m->data[i] = function(0, 1);
	}
}

matrix* matrix_random_uniform(int ROWS, int COLS, double left, double right, bool requires_grad){
	matrix* m = matrix_alloc( ROWS,  COLS, requires_grad);
	for(size_t i = 0; i < m->size; i++){
		m->data[i] = rand_uniform(left, right);
	}
	return m;
}


matrix* matrix_random_normal(int ROWS, int COLS, double mu, double sigma, bool requires_grad){
	matrix* m = matrix_alloc(ROWS, COLS, requires_grad);
	for(size_t i = 0; i < m->size; i++)
		m->data[i] = rand_normal(mu, sigma);
	return m;
}


void matrix_add_rowwise(matrix* mat,  matrix* vec, matrix* out){
	for (size_t i = 0; i < out->rows; i++) {
		for (size_t j = 0; j < out->cols; j++) {
			out->data[i*out->cols + j] = vec->data[j]+mat->data[i*mat->cols + j]; 
		} 
	}
}


void matrix_add_colwise(matrix* mat, matrix* vec, matrix* out){
	for (size_t i = 0; i < out->rows; i++) {
		for (size_t j = 0; j < out->cols; j++) {
			out->data[i*out->cols + j] = vec->data[i]+mat->data[i*mat->cols + j]; 
		} 
	}

}

void matrix_scalar_mul(matrix* input, double scalar, matrix* output){
	for(size_t i = 0; i < input->size; i++){
		output->data[i] = scalar*input->data[i];
	}
}

matrix* matrix_copy(const matrix* input){
	if(MATRIX_NULL(input)){
		MATRIX_ERROR("Invalid matrix argument(s) in matrix_copy()\n");
	}
	matrix* output = matrix_alloc(input->rows, input->cols, input->requires_grad);
	for(size_t i = 0; i < output->size; i++){
		output->data[i] = input->data[i];
	}
	if(input->requires_grad){
		matrix_grad_on(output);
	}
	return output;
}



matrix* matrix_max(const matrix* m){
	if(MATRIX_NULL(m) || *(m->ref_count) == 0){
		MATRIX_ERROR("matrix passed to matrix_max() is NULL or already has been freed.\n");
	}
	matrix* max = matrix_alloc(1,1, 0); 
	max->data[0] = m->data[0];
	for(size_t i = 1; i < m->size; i++){
		if(max->data[0] <= m->data[i]) 
			max->data[0] = m->data[i];
	}
	return max;
}

matrix* matrix_min(const matrix* m){
	if(MATRIX_NULL(m) || *(m->ref_count) == 0){
		MATRIX_ERROR("matrix passed to matrix_max() is NULL or already has been freed.\n");
	}
	matrix* max = matrix_alloc(1,1, 0); 
	max->data[0] = m->data[0];
	for(size_t i = 1; i < m->size; i++){
		if(max->data[0] > m->data[i]) 
			max->data[0] = m->data[i];
	}
	return max;
}

matrix* matrix_mean(const matrix* m){
	matrix* mean = matrix_alloc(1, 1, 0);
	for(size_t i = 0; i < m->size; i++){
		mean->data[0] += m->data[i];
	}
	mean->data[0]  /= m->size;
	return mean;
}


matrix* matrix_std(const matrix* m){
	if(MATRIX_NULL(m) || *(m->ref_count) == 0){
		MATRIX_ERROR("matrix passed to matrix_max() is NULL or already has been freed.\n");
	}
	double mu  = 0;
	for(size_t i = 0; i < m->size; i++){
		mu += m->data[i];
	}
	mu /= m->size;
	matrix* sigma = matrix_alloc(1,1, 0);
	for(size_t i = 0; i < m->size; i++){
		sigma->data[0] = (m->data[i] - mu) * (m->data[i] - mu);
	}
	sigma->data[0] /= m->size;
	sigma->data[0] = sqrt(sigma->data[0]);
	return sigma;
}

matrix* matrix_sum(const matrix* m){
	if(MATRIX_NULL(m) || *(m->ref_count) == 0){
		MATRIX_ERROR("matrix passed to matrix_sum() is NULL or already has been freed.\n");
	}
	// double sum = 0;
	matrix* sum = matrix_alloc(1, 1, 0);
	
	for(size_t i = 0; i < m->size; i++)
		sum->data[0] += m->data[i];
	return sum;
}

matrix* matrix_linspace(double start, double end, size_t num, bool requires_grad){
	if(end < start){
		MATRIX_ERROR("Invalid argument(s) in matrix_linspace(); Ensure end > start\n");
	}
	matrix* out = matrix_alloc(1, num, requires_grad);
	double step = (end - start)/num;
	for(size_t i = 0; i < num; i++){
		out->data[i] = start + i*step; 
	}
	return out;
}

matrix* matrix_arange(double start, double end, double step, bool requires_grad){
	if((end < start) || step <= 0){
		MATRIX_ERROR("Invalid argument(s) in matrix_linspace(); Ensure end > start and step >= 0\n");
	}
	size_t num = (end - start)/step;
	matrix* out = matrix_alloc(1, num, requires_grad);
	for(size_t i = 0; i < num; i++){
		out->data[i] = start + i*step; 
	}
	return out;
}

// TODO
double matrix_det(const matrix* m){
	if(MATRIX_NULL(m))
		MATRIX_ERROR("NULL matrix in matrix_determinant()\n");
	if(m->rows == 2 && m->cols == 2){
		return (get(m, 0, 0)*get(m, 1, 1))-(get(m, 0, 1)*get(m, 1, 0));
	}
	return 0;
}


//	TODO
matrix* matrix_inverse(const matrix* m){
	matrix* out = matrix_alloc(m->rows, m->cols, m->requires_grad);
	if(m->rows == 2 && m->cols == 2){
		set(out, (1/matrix_det(m))*(get(m, 1, 1)), 0, 0) ; 
		set(out, (1/matrix_det(m))*(-get(m, 0, 1)), 0, 1) ; 
		set(out, (1/matrix_det(m))*(-get(m, 1, 0)), 1, 0) ; 
		set(out, (1/matrix_det(m))*(get(m, 0, 0)), 1, 1) ; 
	}
	return out;
}

double matrix_trace(const matrix* m){
	if(!matrix_is_square(m))
		MATRIX_ERROR("Matrix argument is not square in matrix_trace()\n");
	double trace = 0;
	for(size_t i = 0; i < m->rows; i++){
		trace += get(m, i, i);
	}
	return trace;
}


