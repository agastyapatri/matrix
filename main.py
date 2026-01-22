import numpy as np 
x = np.zeros((1000, 1000))
y = np.zeros((1000, 1000))

import timeit
start = timeit.default_timer()
x@y
end = timeit.default_timer()
print(end - start)

