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
    |-functional.h
    |-linalg.h
    |-linalg.c
```



Refer to [the NumPy linear algebra reference](https://numpy.org/doc/stable/reference/routines.linalg.html) for design inspiration.


### **Matrix and vector products** 
### **Decompositions**
### **Matrix Eigenvalues** 
### **Norms and other numbers** 

