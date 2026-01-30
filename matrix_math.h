/*
 *	A library of commonly used math routines and constants
 */ 
#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <math.h>
#include <stdlib.h>
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
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = inp1->data[i] + inp2->data[i];
}

static inline void MATRIX_SUB(matrix* inp1, matrix* inp2, matrix* out){
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = inp1->data[i] - inp2->data[i];
}

static inline void MATRIX_MUL(matrix* inp1, matrix* inp2, matrix* out){
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = inp1->data[i] * inp2->data[i];
}

static inline void MATRIX_DIV(matrix* inp1, matrix* inp2, matrix* out){
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = inp1->data[i] / inp2->data[i];
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







// static inline unary_op get_unary_operation(OPTYPE operation){
// 	switch (operation) {
// 		case(SQUARE):
// 			return matrix_square;
// 		case(CUBE):
// 			return  matrix_cube;
// 		case(LOG):
// 			return matrix_log;
// 		case(EXP):
// 			return matrix_exp;
// 		case(SIN):
// 			return matrix_sin;
// 		case(COS):
// 			return matrix_cos;
// 		case(TAN):
// 			return matrix_tan;
// 		case(ARCSIN):
// 			return matrix_arcsin;
// 		case(ARCCOS):
// 			return matrix_arccos;
// 		case(ARCTAN):
// 			return matrix_arctan;
// 		case(SINH):
// 			return matrix_sinh;
// 		case(COSH):
// 			return matrix_cosh;
// 		case(TANH):
// 			return matrix_tanh;
// 		case(NONE):
// 			return NULL; 
// 		case(ADD):
// 			return NULL; 
// 		case(MUL):
// 			return NULL; 
// 		case(SUB):
// 			return NULL;
// 		case(DIV):
// 			return NULL;
// 		case(MATMUL):
// 			return NULL;
// 	}
// }
//
// static inline binary_op get_binary_operation(OPTYPE operation){
// 	switch (operation) {
// 		case(ADD):
// 			return matrix_add;
// 		case(MUL):
// 			return matrix_mul;
// 		case(SUB):
// 			return matrix_sub;
// 		case(DIV):
// 			return matrix_div;
// 		case(MATMUL):
// 			return NULL;
// 		case(SQUARE):
// 			return NULL;
// 		case(CUBE):
// 			return NULL;
// 		case(LOG):
// 			return NULL;
// 		case(EXP):
// 			return NULL;
// 		case(SIN):
// 			return NULL;
// 		case(COS):
// 			return NULL;
// 		case(TAN):
// 			return NULL;
// 		case(ARCSIN):
// 			return NULL;
// 		case(ARCCOS):
// 			return NULL;
// 		case(ARCTAN):
// 			return NULL;
// 		case(SINH):
// 			return NULL;
// 		case(COSH):
// 			return NULL;
// 		case(TANH):
// 			return NULL;
// 		case(NONE):
// 			return NULL;
// 	}
// }
//
//
static inline char* get_optype_string(OPTYPE op){
	switch (op) {
		case ADD: 
			return "add";
		case SUB: 
			return "sub";
		case MUL: 
			return "mul";
		case DIV: 
			return "div";
		case SQUARE: 
			return "square";
		case CUBE: 
			return "cube";
		case NONE: 
			return "none";
		case SIN:
			return "sin";
		case COS:
			return "sin";
		case TAN:
			return "tan";
		case ARCSIN:
			return "arcsin";
		case ARCCOS:
			return "arccos";
		case ARCTAN:
			return "arctan";
		case SINH:
			return "sinh";
		case COSH:
			return "cosh";
		case TANH: 
			return "tanh";
		case LOG: 
			return "log";
		case EXP: 
			return "exp";
		case MATMUL: 
			return "matmul";
	}
	return NULL;
}

#endif // !matrix_INCLUDE_FUNCTIONAL_H

