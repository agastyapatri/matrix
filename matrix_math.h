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
#define SQRT2 1.414213562373
#define LN2 0.69314718056
#define EPSILON (double)1e-9	

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

static inline void BUF_MATMUL(double* inp1, double* inp2, double* out, size_t inp1rows, size_t inp1cols, size_t inp2cols){
	for(size_t bi = 0; bi < inp1rows; bi+=BLOCK_SIZE){
		for(size_t bk = 0; bk < inp1cols; bk+=BLOCK_SIZE){
			for(size_t bj = 0; bj < inp2cols; bj+=BLOCK_SIZE){
				for(size_t i = bi; (i < inp1rows) && (i < bi + BLOCK_SIZE); i++){



					//	kernel dot prodct loop
					for(size_t k = bk; (k < inp1cols) && (k < bk + BLOCK_SIZE); k++){
						double r = inp1[i*inp1cols + k];
						for(size_t j = bj; (j < inp2cols) && (j < bj + BLOCK_SIZE); j++){
							out[i*inp2cols + j] += r * inp2[k*inp2cols + j];
						}


					} 
				}
			} 
		} 
	}

}

static inline void BUF_POW(double* inp1, double* inp2, double* out, size_t size){
	for(size_t i = 0; i < size; i++){
		out[i] = pow(inp1[i], inp2[i]);
	}
}

static inline void BUF_SIN(double* inp1, double* out, size_t size){
	for(size_t i = 0; i < size; i++){
		out[i] = sin(inp1[i]);
	}
}

static inline void BUF_TANH(double* inp1, double* out, size_t size){
	for(size_t i = 0; i < size; i++){
		out[i] = tanh(inp1[i]);
	}
}

static inline void BUF_COS(double* inp1, double* out, size_t size){
	for(size_t i = 0; i < size; i++){
		out[i] = cos(inp1[i]);
	}
}

static inline void BUF_SIGMOID(double* inp, double* out, size_t size){
	for(size_t i = 0; i < size; i++){
		out[i] = 1.0 / (1 + exp(-inp[i]));
	}
}

static inline void BUF_RELU(double* inp, double* out, size_t size){
	for(size_t i = 0; i < size; i++){
		out[i] = (inp[i] > 0) ? inp[i] : 0;
	}
}

static inline void BUF_LOG(double* inp, double* out, size_t size){
	for(size_t i = 0; i < size; i++){
		out[i] = log(inp[i]);
	}
}

static inline void BUF_EXP(double* inp, double* out, size_t size){
	for(size_t i = 0; i < size; i++){
		out[i] = exp(inp[i]);
	}
}

static inline void BUF_ADD(double* inp1, double* inp2, double* out, size_t rows, size_t cols, size_t stride){
	size_t vector_limit = (cols / 4) * 4;
	for(size_t i = 0; i < rows; i++){
		double* d1 = inp1 + (i * stride);
		double* d2 = inp2 + (i * stride);
		double* o = out + (i * stride);
		size_t j = 0;
		for(; j <= vector_limit; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_add_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < cols; j++){
			o[j] = d1[j] + d2[j];
		}
	}
}

static inline void BUF_SUB(double* inp1, double* inp2, double* out, size_t rows, size_t cols, size_t stride){
	size_t vector_limit = (cols / 4) * 4;
	for(size_t i = 0; i < rows; i++){
		double* d1 = inp1 + (i * stride);
		double* d2 = inp2 + (i * stride);
		double* o = out + (i * stride);
		size_t j = 0;
		for(; j <= vector_limit; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_sub_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < cols; j++){
			o[j] = d1[j] - d2[j];
		}
	}
}

static inline void BUF_MUL(double* inp1, double* inp2, double* out, size_t rows, size_t cols, size_t stride){
	size_t vector_limit = (cols / 4) * 4;
	for(size_t i = 0; i < rows; i++){
		double* d1 = inp1 + (i * stride);
		double* d2 = inp2 + (i * stride);
		double* o = out + (i * stride);
		size_t j = 0;
		for(; j <= vector_limit; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_mul_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < cols; j++){
			o[j] = d1[j] * d2[j];
		}
	}
}

static inline void BUF_DIV(double* inp1, double* inp2, double* out, size_t rows, size_t cols, size_t stride){
	size_t vector_limit = (cols / 4) * 4;
	for(size_t i = 0; i < rows; i++){
		double* d1 = inp1 + (i * stride);
		double* d2 = inp2 + (i * stride);
		double* o = out + (i * stride);
		size_t j = 0;
		for(; j <= vector_limit; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_div_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < cols; j++){
			o[j] = d1[j] + d2[j];
		}
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

static inline double dsigmoid(double x){
	return (1.0 / (1 + exp(-x)))*(1 - (1.0 / (1 + exp(-x)))	);
}

static inline double drelu(double x){
	return (x > 0) ? 1 : 0;

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
		// case DIV: 
		// 	return "div";
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
		case SIGMOID: 
			return "sigmoid";
		case RELU: 
			return "relu";
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
