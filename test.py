import numpy as np 

if __name__ == "__main__":
    x = np.random.randn(1*1024, 1*1024)
    # y = np.random.randn(5*1024, 5*1024)
    import timeit
    start = timeit.default_timer()
    x = x*x
    end = timeit.default_timer()
    print(end - start)
