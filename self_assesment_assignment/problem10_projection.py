import numpy as np
from numpy import linalg

np.set_printoptions(precision=4)

v1 = [1, 2, 3, 4]
v2 = [4,-2, -6, -7]
v3 = [3, 4, -2, 1]
x =  [1, 2, 3, 7]

A = np.transpose(np.vstack([v1,v2,v3]))
x = np.array(x)
Q,R = linalg.qr(A)
x_star = np.dot(x, Q[:,0])*Q[:,0] + np.dot(x, Q[:, 1])*Q[:,1] + np.dot(x, Q[:,2])*Q[:,2]
print (Q)
print (x_star)
x_even = x - x_star
print (x_even)
print (x_even+x_star)
print (np.dot(x_even, x_star))