import numpy as np 

if __name__ == "__main__":
    x = np.random.randn(16, 784)
    y = np.random.randn(784, 392)
    import timeit
    start = timeit.default_timer()
    z = x@y
    end = timeit.default_timer()
    print(end - start)
