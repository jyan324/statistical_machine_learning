import numpy as np
from numpy import linalg as linalg

A = np.array(   [[-2,  2,  2, -2,  0],
                 [-2,  2,  2, -2,  0],
                 [ 2, -2, -2, -2,  0],
                 [ 0,  0,  0,  0,  2],
                 [ 0,  0,  0,  0,  2]],
                dtype=np.float)

U, S, V = linalg.svd (A)

print ("SVD of A completed")
np.set_printoptions(suppress=True)
print ("U:")
print (U)
print ("S:")
print (S)
print ("V:")
print (V)