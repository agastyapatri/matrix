/*
 *	A library of commonly used math routines and constants
 */ 
#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <math.h>
#include <stdlib.h>
#include <immintrin.h>
#include "matrix.h"

#define PI 3.1415926545897932	
#define SQRT2 1.414213562373	// archimedes' constant
#define LN2 0.69314718056	// natural log of 2
#define EPSILON (double)1e-9	// used for error margins in math + log calculations

typedef double (*unary_op)(double);
typedef double (*binary_op)(double, double);
typedef double (*ternary_op)(double, double, double);


static inline void MATRIX_UNARY_OP(matrix* inp1, matrix* out, unary_op function){
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = function(inp1->data[i]);
}
static inline void MATRIX_BINARY_OP(matrix* inp1, matrix* inp2, matrix* out, binary_op function){
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = function(inp1->data[i], inp2->data[i]);
}

static inline void MATRIX_ADD(matrix* inp1, matrix* inp2, matrix* out){
	for(size_t i = 0; i < inp1->rows; i++){
		double* d1 = inp1->data + (i * inp1->stride);
		double* d2 = inp2->data + (i * inp2->stride);
		double* o = out->data + (i * out->stride);
		size_t j = 0;
		for(; j <= inp1->cols - 4; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_add_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < inp1->cols; j++){
			o[j] = d1[j] + d2[j];
		}
	}
}

static inline void MATRIX_SUB(matrix* inp1, matrix* inp2, matrix* out){
	for(size_t i = 0; i < inp1->rows; i++){
		double* d1 = inp1->data + (i * inp1->stride);
		double* d2 = inp2->data + (i * inp2->stride);
		double* o = out->data + (i * out->stride);
		size_t j = 0;
		for(; j <= inp1->cols - 4; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_sub_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < inp1->cols; j++){
			o[j] = d1[j] + d2[j];
		}
	}
}

static inline void MATRIX_MUL(matrix* inp1, matrix* inp2, matrix* out){
	for(size_t i = 0; i < inp1->rows; i++){
		double* d1 = inp1->data + (i * inp1->stride);
		double* d2 = inp2->data + (i * inp2->stride);
		double* o = out->data + (i * out->stride);
		size_t j = 0;
		for(; j <= inp1->cols - 4; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_mul_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < inp1->cols; j++){
			o[j] = d1[j] + d2[j];
		}
	}

}


static inline void MATRIX_DIV(matrix* inp1, matrix* inp2, matrix* out){
	for(size_t i = 0; i < inp1->rows; i++){
		double* d1 = inp1->data + (i * inp1->stride);
		double* d2 = inp2->data + (i * inp2->stride);
		double* o = out->data + (i * out->stride);
		size_t j = 0;
		for(; j <= inp1->cols - 4; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_div_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < inp1->cols; j++){
			o[j] = d1[j] + d2[j];
		}
	}
}

static inline void MATRIX_MATMUL(matrix* inp1, matrix* inp2, matrix* out){
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

static inline void MATRIX_POW(matrix* inp1, matrix* inp2, matrix* out){
	for(size_t i = 0; i < inp1->size; i++){
		out->data[i] = pow(inp1->data[i], inp2->data[i]);
	}
}

static inline void MATRIX_SIN(matrix* inp1, matrix* out){
	for(size_t i = 0; i < inp1->size; i++){
		out->data[i] = sin(inp1->data[i]);
	}
}

static inline void MATRIX_TANH(matrix* inp1, matrix* out){
	for(size_t i = 0; i < inp1->size; i++){
		out->data[i] = tanh(inp1->data[i]);
	}
}

static inline void MATRIX_COS(matrix* inp1, matrix* out){
	for(size_t i = 0; i < inp1->size; i++){
		out->data[i] = cos(inp1->data[i]);
	}
}

static inline void MATRIX_LOG(matrix* inp1, matrix* out){
	for(size_t i = 0; i < inp1->size; i++){
		out->data[i] = log(inp1->data[i]);
	}
}

static inline void MATRIX_EXP(matrix* inp1, matrix* out){
	for(size_t i = 0; i < inp1->size; i++){
		out->data[i] = exp(inp1->data[i]);
	}
}

/*********************************************
 *	SCALAR FUNCTIONS
 *********************************************/ 

static inline double dtanh(double x){
	return 1 - pow(tanh(x), 2);
}

static inline double dsquare(double x){
	return 2*x;
}

static inline double dcube(double x){
	return 3*x*x;
}

static inline double dexp(double x){
	return exp(x);
}

static inline double dlog(double x){
	return 1 / x;
}


static inline double dsin(double x){
	return cos(x);
}

static inline double dcos(double x){
	return -sin(x);
}

static inline double dtan(double x){
	return 1 + pow(tan(x), 2);
}



static inline double rand_double(){
	return rand()/(double)RAND_MAX;
}

static inline double normal(double x, double mu, double sigma){
	double temp = ((x - mu)*(x - mu))/(sigma*sigma);
	return (1/sqrt(2 * PI * sigma))*(exp(-0.5 * temp));
}


static inline double rand_normal(double mu, double sigma){
	double n2 = 0.0; 
	double n2_cached = 0.0; 
	if(!n2_cached){
		double u1 = rand_double();
		double u2 = rand_double();
		double r = sqrt(-2.0 * log(u1));
		double theta = 2 * PI * u2;
		n2 = r * sin(theta);
		n2_cached = 1;
		return r * cos(theta) * sigma + mu;
	}
	else{
		n2_cached = 0;
		return n2*sigma + mu;
	}

}


static inline double rand_uniform(double left, double right){
	return rand_double()*(right - left) + left;
}


static inline double squared_error(double x, double y){
	return (x - y)*(x - y);
}


static inline char* get_optype_string(OPTYPE op){
	switch (op) {
		case ADD: 
			return "add";
		case POW: 
			return "pow";
		case SUB: 
			return "sub";
		case MUL: 
			return "mul";
		case DIV: 
			return "div";
		case NONE: 
			return "none";
		case SIN:
			return "sin";
		case COS:
			return "cos";
		case LOG: 
			return "log";
		case EXP: 
			return "exp";
		case MATMUL: 
			return "matmul";
		case TANH: 
			return "tanh";
		// case SQUARE: 
		// 	return "square";
		// case CUBE: 
		// 	return "cube";
		// case TAN:
		// 	return "tan";
		// case ARCSIN:
		// 	return "arcsin";
		// case ARCCOS:
		// 	return "arccos";
		// case ARCTAN:
		// 	return "arctan";
		// case SINH:
		// 	return "sinh";
		// case COSH:
		// 	return "cosh";
	}
	return NULL;
}






#endif // !MATRIX_MATH_H

