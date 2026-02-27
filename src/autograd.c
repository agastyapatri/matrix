#include "autograd.h"
#include "matrix.h"
#include "matrix_math.h"
void matrix_one_grad(matrix* out){
	for(size_t i = 0; i < out->size; i++)
		out->grad[i] = 1.0;
}

void matrix_zero_grad(matrix* out){
	for(size_t i = 0; i < out->size; i++)
		out->grad[i] = 0.0;
}

void ad_mul_backward(matrix* out){
	for(size_t i = 0; i < out->rows; i++){
		double* ogradrow = out->grad + (i * out->stride);
		double* p0datarow = out->previous[0]->data + (i * out->stride);
		double* p1datarow = out->previous[1]->data + (i * out->stride);
		double* p0gradrow = out->previous[0]->grad + (i * out->stride);
		double* p1gradrow = out->previous[1]->grad + (i * out->stride);

		for(size_t j = 0; j < out->cols; j++){
			p0gradrow[j] += ogradrow[j] * p1datarow[j];
			p1gradrow[j] += ogradrow[j] * p0datarow[j];
		}
	}
}
void ad_add_backward(matrix* out){
	for(size_t i = 0; i < out->rows; i++){
		double* ogradrow = out->grad + (i * out->stride);
		double* p0gradrow = out->previous[0]->grad + (i * out->stride);
		double* p1gradrow = out->previous[1]->grad + (i * out->stride);
		for(size_t j = 0; j < out->cols; j++){
			p0gradrow[j] += ogradrow[j];
			p1gradrow[j] += ogradrow[j];
		}
	}
}
void ad_sub_backward(matrix* out){
	for(size_t i = 0; i < out->rows; i++){
		double* ogradrow = out->grad + (i * out->stride);
		double* p0gradrow = out->previous[0]->grad + (i * out->stride);
		double* p1gradrow = out->previous[1]->grad + (i * out->stride);
		for(size_t j = 0; j < out->cols; j++){
			p0gradrow[j] += ogradrow[j];
			p1gradrow[j] -= ogradrow[j];
		}
	}
}

void ad_log_backward(matrix* out){
	for(size_t i = 0; i < out->rows; i++){
		double* ogradrow = out->grad + (i * out->stride);
		double* p0datarow = out->previous[0]->data + (i * out->stride);
		double* p0gradrow = out->previous[0]->grad + (i * out->stride);
		for(size_t j = 0; j < out->cols; j++){
			p0gradrow[j] += ogradrow[j] * dlog(p0datarow[j]);
		}
	}

}

void ad_exp_backward(matrix* out){
	for(size_t i = 0; i < out->rows; i++){
		double* ogradrow = out->grad + (i * out->stride);
		double* p0datarow = out->previous[0]->data + (i * out->stride);
		double* p0gradrow = out->previous[0]->grad + (i * out->stride);
		for(size_t j = 0; j < out->cols; j++){
			p0gradrow[j] += ogradrow[j] * dexp(p0datarow[j]);
		}
	}

}

void ad_sin_backward(matrix* out){
	for(size_t i = 0; i < out->rows; i++){
		double* ogradrow = out->grad + (i * out->stride);
		double* p0gradrow = out->previous[0]->grad + (i * out->stride);
		double* p0datarow = out->previous[0]->data + (i * out->stride);
		for(size_t j = 0; j < out->cols; j++){
			p0gradrow[j] += ogradrow[j] * dsin(p0datarow[j]);
		}
	}

}

void ad_tanh_backward(matrix* out){
	for(size_t i = 0; i < out->rows; i++){
		double* ogradrow = out->grad + (i * out->stride);
		double* p0datarow = out->previous[0]->data + (i * out->stride);
		double* p0gradrow = out->previous[0]->grad + (i * out->stride);
		for(size_t j = 0; j < out->cols; j++){
			p0gradrow[j] += ogradrow[j] * dtanh(p0datarow[j]);
		}
	}
}

void ad_cos_backward(matrix* out){
	for(size_t i = 0; i < out->rows; i++){
		double* ogradrow = out->grad + (i * out->stride);
		double* p0datarow = out->previous[0]->data + (i * out->stride);
		double* p0gradrow = out->previous[0]->grad + (i * out->stride);
		for(size_t j = 0; j < out->cols; j++){
			p0gradrow[j] += ogradrow[j] * dcos(p0datarow[j]);
		}
	}
}

void ad_sigmoid_backward(matrix* out){
	for(size_t i = 0; i < out->rows; i++){
		double* ogradrow = out->grad + (i * out->stride);
		double* p0datarow = out->previous[0]->data + (i * out->stride);
		double* p0gradrow = out->previous[0]->grad + (i * out->stride);
		for(size_t j = 0; j < out->cols; j++){
			p0gradrow[j] += ogradrow[j] * dsigmoid(p0datarow[j]);
		}
	}
}

void ad_relu_backward(matrix* out){
	for(size_t i = 0; i < out->rows; i++){
		double* ogradrow = out->grad + (i * out->stride);
		double* p0datarow = out->previous[0]->data + (i * out->stride);
		double* p0gradrow = out->previous[0]->grad + (i * out->stride);
		for(size_t j = 0; j < out->cols; j++){
			p0gradrow[j] += ogradrow[j] * drelu(p0datarow[j]);
		}
	}
}

void ad_sum_backward(matrix* out){
	double ograd = *out->grad;
	for(size_t i = 0; i < out->previous[0]->rows; i++){
		double* p0gradrow = out->previous[0]->grad + (i * out->previous[0]->stride);
		for(size_t j = 0; j < out->previous[0]->cols; j++){
			p0gradrow[j] += ograd;
		}
	}
}

void ad_mean_backward(matrix* out){
	double ograd = *out->grad;
	int n = (out->previous[0]->rows * out->previous[0]->cols);
	for(size_t i = 0; i < out->previous[0]->rows; i++){
		double* p0gradrow = out->previous[0]->grad + (i * out->previous[0]->stride);
		for(size_t j = 0; j < out->previous[0]->cols; j++){
			p0gradrow[j] += (1.0/n) * ograd;
		}
	}
}

void ad_std_backward(matrix* out){
	double ograd = *out->grad;
	double odata = *out->data;
	double mean = 0; 
	BUF_MEAN(out->previous[0]->data, &mean, out->previous[0]->rows, out->previous[0]->cols, out->previous[0]->stride);
	int n = (out->previous[0]->rows * out->previous[0]->cols);
	for(size_t i = 0; i < out->previous[0]->rows; i++){
		double* p0gradrow = out->previous[0]->grad + (i * out->previous[0]->stride);
		double* p0datarow = out->previous[0]->data + (i * out->previous[0]->stride);
		for(size_t j = 0; j < out->previous[0]->cols; j++){
			p0gradrow[j] += ograd * (1.0 / (n * odata)) * (p0datarow[j] - mean);
		}
	}

}


void ad_matmul_backward(matrix* out){
	matrix* inp0 = out->previous[0];
	matrix* inp1 = out->previous[1];
	matrix* inp0transpose = matrix_transpose(inp0); // inp0->cols x inp0->rows
	matrix* inp1transpose = matrix_transpose(inp1); // inp1->cols x inp1->rows
	matrix* outgrad = matrix_from_raw(out->grad, out->rows, out->cols); // out->rows x out->cols == inp0->rows x inp1->cols
	matrix* dinp0 = matrix_matmul(outgrad, inp1transpose); // inp0->rows x inp1->rows 
	matrix* dinp1 = matrix_matmul(inp0transpose, outgrad); // inp0->cols x inp1->cols 
	for(size_t i = 0; i < inp0->rows; i++){
		double* inp0row = inp0->grad + (i * inp0->stride);
		double* dinp0row = dinp0->data + (i * dinp0->stride);
		for(size_t j = 0; j < inp0->cols; j++){
			inp0row[j] += dinp0row[j];
		}
	}
	for(size_t i = 0; i < inp1->rows; i++){
		double* inp1row = inp1->grad + (i * inp1->stride);
		double* dinp1row = dinp1->data + (i * dinp1->stride);
		for(size_t j = 0; j < inp1->cols; j++){
			inp1row[j] += dinp1row[j];
		}
	}
	free(outgrad->ref_count);
	matrix_free(outgrad);
	matrix_free(inp0transpose);
	matrix_free(inp1transpose);
	matrix_free(dinp0);
	matrix_free(dinp1);
}

void ad_mse_backward(matrix* out){
	size_t n = out->previous[0]->rows * out->previous[0]->cols;
	double ograd = *out->grad;
	for(size_t i = 0; i < out->previous[0]->rows; i++){
		double* p0gradrow = out->previous[0]->grad + (i * out->previous[0]->stride);
		double* p0datarow = out->previous[0]->data + (i * out->previous[0]->stride);
		double* labeldatarow = out->previous[1]->data + (i * out->previous[1]->stride);
		for(size_t j = 0; j < out->previous[0]->cols; j++){
			p0gradrow[j] += ograd * (2.0 / n) * (p0datarow[j] - labeldatarow[j]);
		}
	}
}

void ad_mae_backward(matrix* out){
	size_t n = out->previous[0]->rows * out->previous[0]->cols;
	double ograd = *out->grad;
	for(size_t i = 0; i < out->previous[0]->rows; i++){
		double* p0gradrow = out->previous[0]->grad + (i * out->previous[0]->stride);
		double* p0datarow = out->previous[0]->data + (i * out->previous[0]->stride);
		double* labeldatarow = out->previous[1]->data + (i * out->previous[1]->stride);
		for(size_t j = 0; j < out->previous[0]->cols; j++){
			p0gradrow[j] += ograd * (1.0 / n)*((p0datarow[j] < labeldatarow[j]) ? -1 : 1);
		}
	}
}

void matrix_grad(matrix* out){
	matrix_one_grad(out);
	switch (out->op) {
		case ADD:
			ad_add_backward(out);
			return;
		case SUB:
			ad_add_backward(out);
			return;
		case MUL:
			ad_mul_backward(out);
			return; 
		case LOG: 
			ad_log_backward(out);
			return;
		case EXP: 
			ad_exp_backward(out);
			return;
		case TANH: 
			ad_tanh_backward(out);
			return;
		case SIN: 
			ad_sin_backward(out);
			return;
		case COS: 
			ad_cos_backward(out);
			return;
		case SIGMOID: 
			ad_sigmoid_backward(out);
			return;
		case RELU: 
			ad_relu_backward(out);
			return;
		case MATMUL: 
			ad_matmul_backward(out);
			return;
		case SUM: 
			ad_sum_backward(out);
			return;
		case MSE: 
			ad_mse_backward(out);
			return;
		case MAE: 
			ad_mae_backward(out);
			return;
		case MEAN: 
			ad_mean_backward(out);
			return;
		case STD: 
			ad_std_backward(out);
			return;
		case NONE: 
			return;
	}
} 




// void ad_matmul_backward_alt(matrix* out){
// 	size_t P = out->previous[0]->rows; 
// 	size_t Q = out->previous[0]->cols; 
// 	size_t R = out->previous[1]->cols; 
// 	double* inp1grad = out->previous[0]->grad;
// 	double* inp2grad = out->previous[1]->grad;
// 	double* outgrad = out->grad;
// 	int inp1stride = out->previous[0]->stride;
// 	int inp2stride = out->previous[1]->stride;
// 	int outstride = out->stride;
//
//
// 	// matmul of outgrad and inp2transpose (P, R) x (R, Q) = (P, Q)
// 	double dinp1[P * Q];
// 	memset(dinp1, 0, P*Q*sizeof(double));
// 	for(size_t i = 0; i < P; i++){
// 		double* outrow = outgrad + (i * outstride);
// 		double* dinp1row = dinp1 + (i * inp1stride);
// 		for(size_t k = 0; k < Q; k++){
// 			double outgrad_ik = outrow[k];
// 			double* inp2row = inp2grad + (k * inp2stride);
// 			for(size_t j = 0; j < R; j++){
// 				dinp1row[j] += outgrad_ik * inp2row[j];
// 			}
// 		} 
// 	}
//
// 	double dinp2[Q*R]; 
// 	memset(dinp1, 0, P*Q*sizeof(double));
// }
//
