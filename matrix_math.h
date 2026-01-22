/*
 *	A library of commonly used math routines and constants
 */ 
#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <math.h>
#include <stdlib.h>

#define PI 3.1415926545897932	
#define SQRT2 1.414213562373	// archimedes' constant
#define LN2 0.69314718056	// natural log of 2
#define EPSILON (double)1e-9	// used for error margins in math + log calculations
								//
typedef enum {
	NONE,
	ADD, 
	SUB,
	MUL,
	DIV,
	// SQUARE,
	// CUBE,
	// MATMUL,
	// MEAN,
	// RELU,
	// SIGMOID,
} OPTYPE;
char* get_optype_string(OPTYPE op);


/*********************************************
 *	FUNCTIONS
 *********************************************/ 

static inline double matrix_add(double x, double y){
	return x + y;
}

static inline double matrix_sub(double x, double y){
	return x - y;
}

static inline double matrix_mul(double x, double y){
	return x * y;
}

static inline double matrix_div(double x, double y){
	return x / y;
}

static inline double matrix_square(double x){
	return x*x;
}

static inline double matrix_cube(double x){
	return x*x*x;
}

static inline double matrix_sigmoid(double x){
	return 1.0 / (1 + exp(-x));
}

static inline double matrix_relu(double x){
	return (x > 0) ? x : 0.0;
}

static inline double matrix_tanh(double x){
	return tanh(x);
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
static inline double matrix_dsigmoid(double x){
	return matrix_sigmoid(x)*(1 - matrix_sigmoid(x));
}


static inline double matrix_drelu(double x){
	return (x > 0) ? 1.0 : 0.0;
}

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


// static inline double matrix_arcsin(double x){
// 	return asin(x);
// }
//
// static inline double matrix_arccos(double x){
// 	return acos(x);
// }
//
// static inline double matrix_arctan(double x){
// 	return atan(x);
// }


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








#endif // !matrix_INCLUDE_FUNCTIONAL_H

