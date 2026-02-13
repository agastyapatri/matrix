#include "autograd.h"
#include "matrix_math.h"
void matrix_one_grad(matrix* out){
	for(size_t i = 0; i < out->size; i++)
		out->grad[i] = 1.0;
}

void matrix_zero_grad(matrix* out){
	for(size_t i = 0; i < out->size; i++)
		out->grad[i] = 0.0;
}

void matrix_grad(matrix* out){
	int prev0len = out->previous[0]->size;
	double prev0temp[prev0len];
	int prev1len = 0;
	if(out->num_prevs == 2){
		prev1len = out->previous[1]->size;
	}
	double prev1temp[prev1len];

	switch (out->op) {
		case ADD: 
			BUF_ADD(out->previous[0]->grad, out->grad, out->previous[0]->grad, out->rows, out->cols, out->stride);
			BUF_ADD(out->previous[1]->grad, out->grad, out->previous[1]->grad, out->rows, out->cols, out->stride);
			return; 
		case SUB: 
			BUF_ADD(out->previous[0]->grad, out->grad, out->previous[0]->grad, out->rows, out->cols, out->stride);
			BUF_SUB(out->previous[1]->grad, out->grad, out->previous[1]->grad, out->rows, out->cols, out->stride);
			return; 
		case MUL: 
			BUF_MUL(out->grad, out->previous[1]->data, prev0temp, out->rows, out->cols, out->stride);
			BUF_ADD(out->previous[0]->grad, prev0temp, out->previous[0]->grad, out->rows, out->cols, out->stride);
			// BUF_MUL(out->grad, out->previous[0]->data, prev1temp, out->rows, out->cols, out->stride);
			// BUF_ADD(out->previous[1]->grad, prev1temp, out->previous[1]->grad, out->rows, out->cols, out->stride);
			return; 
		case SIN:
			for(int i = 0; i < prev0len; i++)
				prev0temp[i] = dsin(out->previous[0]->data[i]);
			BUF_MUL(out->grad, prev0temp, prev0temp, out->rows, out->cols, out->stride);
			BUF_ADD(out->previous[0]->grad, prev0temp, out->previous[0]->grad, out->rows, out->cols, out->stride);
			return; 
		case COS:
			for(int i = 0; i < prev0len; i++)
				prev0temp[i] = dcos(out->previous[0]->data[i]);
			BUF_MUL(out->grad, prev0temp, prev0temp, out->rows, out->cols, out->stride);
			BUF_ADD(out->previous[0]->grad, prev0temp, out->previous[0]->grad, out->rows, out->cols, out->stride);
			return;
		case LOG: 
			for(int i = 0; i < prev0len; i++)
				prev0temp[i] = dlog(out->previous[0]->data[i]);
			BUF_MUL(out->grad, prev0temp, prev0temp, out->rows, out->cols, out->stride);
			BUF_ADD(out->previous[0]->grad, prev0temp, out->previous[0]->grad, out->rows, out->cols, out->stride);
			return;
		case EXP: 
			for(int i = 0; i < prev0len; i++)
				prev0temp[i] = out->previous[0]->data[i];
			BUF_MUL(out->grad, prev0temp, prev0temp, out->rows, out->cols, out->stride);
			BUF_ADD(out->previous[0]->grad, prev0temp, out->previous[0]->grad, out->rows, out->cols, out->stride);
			return;
		case TANH: 
			for(int i = 0; i < prev0len; i++)
				prev0temp[i] = dtanh(out->previous[0]->data[i]);
			BUF_MUL(out->grad, prev0temp, prev0temp, out->rows, out->cols, out->stride);
			BUF_ADD(out->previous[0]->grad, prev0temp, out->previous[0]->grad, out->rows, out->cols, out->stride);
			return; 
		case SIGMOID: 
			for(int i = 0; i < prev0len; i++)
				prev0temp[i] = out->data[i]*(1 - out->data[i]);
			BUF_MUL(out->grad, prev0temp, prev0temp, out->rows, out->cols, out->stride);
			BUF_ADD(out->previous[0]->grad, prev0temp, out->previous[0]->grad, out->rows, out->cols, out->stride);
			return ;
		case RELU: 
			for(int i = 0; i < prev0len; i++)
				prev0temp[i] = drelu(out->data[i]);
			BUF_MUL(out->grad, prev0temp, prev0temp, out->rows, out->cols, out->stride);
			BUF_ADD(out->previous[0]->grad, prev0temp, out->previous[0]->grad, out->rows, out->cols, out->stride);
			return;
		case NONE: 
			return;
	}
}
