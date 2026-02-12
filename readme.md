#   **2D Linear Algebra** 
A tiny matrix + linalg library from scratch, in C.
This has been my attempt to recreate a small subset of the functionality in NumPy; mostly for self-edification.

At the core of the library lies the `struct matrix`:

```c
typedef struct matrix{
	size_t rows, cols;
	size_t size;
	int* ref_count;
	double *data; 
} matrix;
```

The structure of this project is:
```
matrix/
    |-matrix.h
    |-matrix.c
    |-matrix_math.h
    |-linalg.h  // under construction!   
    |-linalg.c  // under construction!     
```



Refer to [the NumPy linear algebra reference](https://numpy.org/doc/stable/reference/routines.linalg.html) for design inspiration.


##  **Roadmap** 
The next steps are to build out `linalg.h` with more abstracted linear algebra. 

In no particular order:

-   **Parallelization of core math utils**
-   Matrix decompositions
-   Matrix determinant
-   Matrix eigenvalues
-   Gaussian elimination
-   Matrix inverse
-   Matrix adjoint
-   Matrix conjugate
-   Solving systems of linear equations
-   Solving systems of linear equations
-   Sparse Matrices
-   Transforms; DFT, FFT
-   Matrix trace 
-   Logical operations on matrices 
-   Matrix slicing + advanced indexing
-   Matrix I/O; marshalling and unmarshalling 
-   Matrix sorting, searching etc.
-   Parallelizing the underlying routines with SIMD + memory alignment. 
-   Compatibility with NumPy(?)

### Notes on the Development of this Library

1.  Each public facing `matrix* matrix_<function>(matrix*, matrix*)` calls an underlying kernel defined by `void MATRIX_<FUNCTION>(matrix* inp1, matrix* inp2, matrix* out)`. The kernel does the actual computation between the two matrices. This needs to be taken a step further, where the kernel will only operate on the data buffers, and the public API will handle every other meta-operation. The new kernel signature should be `void BUF_<OPERATION>(double* inp1, double* inp2, double* out, int stride, int padding, int alignement, ...)`

2.  Something is going wrong with my interpretation of SIMD strides and operations; I need to go back to the drawing board on vectorization.





##  **Building and using this library** 
I do not recommended using this library (yet) for anything even remotely performant or stable. There are many wrinkles waiting to be ironed out, including a more robust testing framework. Using some BLAS / LAPACK descendant is always going to be the better option. For simple hobbyist code, however, the API is simple enough to get up and running quickly.

```
clang -std=c17 -Wall -Wextra -O3 -c matrix.c -o matrix.o 
ar rcs libmatrix.a matrix.o 
rm matrix.o
```
Do not forget to add `-Ipath/to/matrix.h` in your project LSP settings.














