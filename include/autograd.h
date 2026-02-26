#ifndef AUTOGRAD_H
#define AUTOGRAD_H
#include "matrix_math.h"
#include "matrix.h"
#include <string.h>

#ifndef GRAPH_POPULATION
#define GRAPH_POPULATION 32;
#endif

void ad_matrix_one_grad(matrix* out);
void matrix_zero_grad(matrix* out);

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
