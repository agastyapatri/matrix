#ifndef AUTOGRAD_H
#define AUTOGRAD_H
#include "matrix_math.h"
#include "matrix.h"
#include <string.h>

#ifndef GRAPH_POPULATION
#define GRAPH_POPULATION 32;
#endif

void matrix_one_grad(matrix* out);
void matrix_zero_grad(matrix* out);


static inline void ag_buf_add(double* inp1, double* inp2, double* out, size_t len){
	for(size_t i = 0; i < len; i++){
		out[i] = inp1[i] + inp2[i];
	}
}
static inline void ag_buf_sub(double* inp1, double* inp2, double* out, size_t len){
	for(size_t i = 0; i < len; i++){
		out[i] = inp1[i] - inp2[i];
	}

}
static inline void ag_buf_mul(double* inp1, double* inp2, double* out, size_t len){
	for(size_t i = 0; i < len; i++){
		out[i] = inp1[i] * inp2[i];
	}
}
static inline void ag_buf_div(double* inp1, double* inp2, double* out, size_t len){
	for(size_t i = 0; i < len; i++){
		out[i] = inp1[i] / inp2[i];
	}
}


typedef struct graph {
	matrix** nodes;
	int* ref_count;
	size_t num_nodes;
} graph;
graph* graph_init();
void graph_free(graph* tape);


//	TODO
void matrix_grad(matrix* out);
void matrix_backward(matrix* out);



#endif /* ifndef AUTOGRAD_H
 */
