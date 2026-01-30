#   **Mathematics**
### **Autograd with `matrix`**
This library was built for simple neural network training. 
I need to refactor some code to allow for automatic differentiation. 

The core workflow is: 

1.  Operation recording (Forward pass): every mathematical involving two source matrices must do two things: compute the result and link the result to its parents in the `matrix->previous` array.
2.  Topological Sort: Before backprop can begin, the graph must be sorted so that a child node is always processed before its parent.
3.  Apply the chain rule recursively.





##  **Determinant Calculation** 
-   For 2x2 and 3x3 matrices, use the hardcoded formulae of cofactor expansion
-   For other random shapes, use **LU Decomposition** 

-   The determinant of triangular matrix is simply the product of its diagonal elements. If a matrix can be converted to some composition of a set of triangular matrices, its determinant will also be some composition of the determinants of its constitutent triangular matrices.
-   Through gaussian elimination, with partial pivoting, any square matrix A can be decomposed into 
$$PA = LU$$.




##  **What needs to be done**
1.  `struct matrix` has been modified such that the data and gradient buffers are aligned to 32 bytes - 4 doubles.
2.  `matrix_math.h` needs a complete refactor. All arithmetic must be done to the tune of `matrix* operation(matrix* in1, matrix* in2, matrix* out)`. 
3.  `MATRIX_ADD`, `MATRIX_SUB`, `MATRIX_DIV`, `MATRIX_MUL` all need to be vectorized with AVX2  
4.  `matrix_unary_op` and `matrix_binary_op` need to be rethought. 
5.  `matmul` needs to be vectorized with AVX2 - `_mm256_fmadd_pd` and so forth.
6.  Understand tiled matmul.
7.  Understand how tiling and vectorization of code is done in the abstract. What to look for? What is the CPU doing?
8.  Write more complex benchmarking tools -- re-write `MATRIX_TIMER()` to average over 10 iterations instead of just one. 




