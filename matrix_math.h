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





/*********************************************
 *	FUNCTIONS
 *********************************************/ 

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


// static inline double matrix_add(double x, double y){
// 	return x + y;
// }

// static inline double matrix_sub(double x, double y){
// 	return x - y;
// }
// static inline double matrix_mul(double x, double y){
// 	return x * y;
// }

// static inline double matrix_div(double x, double y){
// 	return x / y;
// }

static inline double matrix_square(double x){
	return x*x;
}

static inline double matrix_cube(double x){
	return x*x*x;
}

// static inline double matrix_sigmoid(double x){
// 	return 1.0 / (1 + exp(-x));
// }

// static inline double matrix_relu(double x){
// 	return (x > 0) ? x : 0.0;
// }

static inline double matrix_tanh(double x){
	return tanh(x);
}
static inline double matrix_sinh(double x){
	return sinh(x);
}
static inline double matrix_cosh(double x){
	return cosh(x);
}


static inline double matrix_sin(double x){
	return sin(x);
}

static inline double matrix_cos(double x){
	return cos(x);
}

static inline double matrix_tan(double x){
	return tan(x);
}
static inline double matrix_arcsin(double x){
	return asin(x);
}

static inline double matrix_arccos(double x){
	return acos(x);
}

static inline double matrix_arctan(double x){
	return atan(x);
}


static inline double matrix_exp(double x){
	return exp(x);
}

static inline double matrix_log(double x){
	return log(x);
}



/*********************************************
 *	DERIVATIVES 
 *********************************************/ 

static inline double matrix_dtanh(double x){
	return 1 - matrix_square(matrix_tanh(x));
}


static inline double matrix_dsquare(double x){
	return 2*x;
}

static inline double matrix_dcube(double x){
	return 3*x*x;
}

static inline double matrix_dexp(double x){
	return exp(x);
}

static inline double matrix_dlog(double x){
	return 1 / x;
}


static inline double matrix_dsin(double x){
	return cos(x);
}

static inline double matrix_dcos(double x){
	return -sin(x);
}

static inline double matrix_dtan(double x){
	return 1 + matrix_square(tan(x));
}



/*
 *	RANDOM NUMBERS AND PROBABILITY DISTRIBUTIONS
 */ 

static inline double rand_double(){
	return rand()/(double)RAND_MAX;
}

static inline double matrix_normal(double x, double mu, double sigma){
	double temp = ((x - mu)*(x - mu))/(sigma*sigma);
	return (1/sqrt(2 * PI * sigma))*(exp(-0.5 * temp));
}


static inline double matrix_rand_normal(double mu, double sigma){
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


static inline double matrix_rand_uniform(double left, double right){
	return rand_double()*(right - left) + left;
}


static inline double matrix_squared_error(double x, double y){
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
// static inline char* get_optype_string(OPTYPE op){
// 	switch (op) {
// 		case ADD: 
// 			return "add";
// 		case SUB: 
// 			return "sub";
// 		case MUL: 
// 			return "mul";
// 		case DIV: 
// 			return "div";
// 		case SQUARE: 
// 			return "square";
// 		case CUBE: 
// 			return "cube";
// 		case NONE: 
// 			return "none";
// 		case SIN:
// 			return "sin";
// 		case COS:
// 			return "sin";
// 		case TAN:
// 			return "tan";
// 		case ARCSIN:
// 			return "arcsin";
// 		case ARCCOS:
// 			return "arccos";
// 		case ARCTAN:
// 			return "arctan";
// 		case SINH:
// 			return "sinh";
// 		case COSH:
// 			return "cosh";
// 		case TANH: 
// 			return "tanh";
// 		case LOG: 
// 			return "log";
// 		case EXP: 
// 			return "exp";
// 		case MATMUL: 
// 			return "matmul";
// 	}
// 	return NULL;
// }

#endif // !matrix_INCLUDE_FUNCTIONAL_H

