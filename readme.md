#   **matrix**
A tiny, self-contained matrix valued autodiff library from scratch, in C.
This has been my attempt to recreate a small subset of the functionality in PyTorch; mostly for self-edification.

At the core of the library lies the `struct matrix`:

```c
typedef struct matrix{
	size_t rows;
	size_t cols;
	int* ref_count;
	double* data; 
	size_t bytes;
	int stride;
	size_t size;
	int padding;
	bool requires_grad;
	double* grad;
	OPTYPE op;
	struct matrix* previous[MAX_PREVS];
	int num_prevs;
} matrix;
```
The purpose of this library is to provide primitives on which neural network abstractions can be defined. Central to that effort is the define-by-run automatic differentiation mechanism. Any computation is a collection of sequential operations whose forward and backward passes are pre-defined. 

An example of the calculation of a derivative:

```c 
matrix* m1 = matrix_random_normal(NUM_ROWS, NUM_COLS, MU, SIGMA, REQUIRES_GRAD);
matrix* m2 = matrix_uniform(NUM_ROWS, NUM_COLS, LEFT, RIGHT, REQUIRES_GRAD);
matrix* m3 = matrix_sin(m1);
matrix* m4 = matrix_cos(m2);


matrix* m5 = matrix_add(m3, m4); //y = sin(x1) + cos(x2) 
matrix_backward(m5) // calcuating the derivative of y wrt (m1, m2, m3, m4)
```





##  **Building and using this library** 
I do not recommended using this library (yet) for anything even remotely performant or stable. There are many wrinkles waiting to be ironed out, including a more robust testing framework. Using some BLAS / LAPACK descendant is always going to be the better option. For simple hobbyist code, however, the API is simple enough to get up and running quickly.

```make
clang -std=c17 -Wall -Wextra -O3 -ffast-math -mavx -mfma -Iinclude/ -c src/autograd.c bin/autograd.o 
clang -std=c17 -Wall -Wextra -O3 -ffast-math -mavx -mfma -Iinclude/ -c src/matrix.c bin/matrix.o 
clang -std=c17 -Wall -Wextra -O3 -ffast-math -Iinclude/ -c main.c bin/main.o 
clang -std=c17 -Wall -Wextra bin/main.o bin/autograd.o bin/matrix.o -o main -lm 
```
Do not forget to add `-Ipath/to/matrix.h` in your project LSP settings :)














