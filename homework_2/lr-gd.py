import numpy as np
import pdb
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets 
from math import exp

# the logistic function
def logistic_func(theta, x):
    t = x.dot(theta)
    g = np.zeros(t.shape)
    # split into positive and negative to improve stability
    g[t>=0.0] = 1.0 / (1.0 + np.exp(-t[t>=0.0])) 
    g[t<0.0] = np.exp(t[t<0.0]) / (np.exp(t[t<0.0])+1.0)
    return g

# function to compute log-likelihood
def neg_log_like(theta, x, y):
    g = logistic_func(theta,x)
    return -sum(np.log(g[y>0.5])) - sum(np.log(1-g[y<0.5]))

# function to compute the gradient of the negative log-likelihood
def log_grad(theta, x, y):
    g = logistic_func(theta,x)
    return -x.T.dot(y-g)
    
# function to compute the Hessian of the log-likelihood
def log_hessian(theta, x):
    dim = x.shape[1]
    g = logistic_func(theta, x)
    H = np.zeros((dim,dim))
    for i in range(x.shape[0]):
        xi = x[i, :].reshape((x.shape[1],1))
        H = H + (np.multiply(xi, np.transpose(xi)))*g[i]*(1-g[i])
    return H

# implementation of gradient descent for logistic regression
def grad_desc(theta, x, y, alpha, tol, maxiter):
    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0*tol
    iter = 0
    while (nll_delta > tol) and (iter < maxiter):
        theta = theta - (alpha * log_grad(theta, x, y))
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta = nll_vec[-2]-nll_vec[-1]
        iter += 1
    return theta, np.array(nll_vec)
 
# implementation of Netwon method based optimization for logistic regression
def newton_opt(theta, x, y, tol, maxiter):
    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0*tol
    iter = 0
    while (nll_delta > tol) and (iter < maxiter):
        alpha = np.linalg.inv(log_hessian(theta, x))
        theta = theta - alpha@log_grad(theta, x, y)
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta = nll_vec[-2]-nll_vec[-1]
        iter += 1
    return theta, np.array(nll_vec)

# implementation of gradient descent for logistic regression
def sgd(theta, x, y, alpha, tol, maxiter):
    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0*tol
    indexes = np.random.permutation(np.arange(0,100))
    max_index = x.shape[0]
    nos = 10000
    print ("Alpha: %f\n"%alpha)
    iter = 0
    while (nll_delta > tol) and (iter < maxiter):
        idx = indexes[(iter%max_index)]
        theta = theta - (alpha * log_grad(theta, x[idx,:], y[idx])) 
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta = nll_vec[-2]-nll_vec[-1]
        iter += 1
    return theta, np.array(nll_vec)


# function to compute output of LR classifier
def lr_predict(theta,x):
    # form Xtilde for prediction
    shape = x.shape
    Xtilde = np.zeros((shape[0],shape[1]+1))
    Xtilde[:,0] = np.ones(shape[0])
    Xtilde[:,1:] = x
    return logistic_func(theta,Xtilde)

## Generate dataset    
np.random.seed(2017) # Set random seed so results are repeatable
x,y = datasets.make_blobs(n_samples=100000,n_features=2,centers=2,cluster_std=6.0)

## build classifier
# form Xtilde
shape = x.shape
xtilde = np.zeros((shape[0],shape[1]+1))
xtilde[:,0] = np.ones(shape[0])
xtilde[:,1:] = x

# Initialize theta to zero
theta = np.zeros(shape[1]+1)

# Run gradient descent
alpha = 1e-6
tol = 1e-3
maxiter = 30000#50000
start = time.time()
# log_hessian(theta, xtilde)
# theta,cost = grad_desc(theta,xtilde,y,alpha,tol,maxiter)
# theta,cost = newton_opt(theta,xtilde, y, tol,maxiter)
theta,cost = sgd(theta,xtilde,y,alpha,tol,maxiter)
end = time.time() - start
print ("\t\tReport of Gradient Descent\n")
print ("Iterations for Convergence: %d \n"%(len(cost)))
print ("Time: %f seconds "%(end))

## Plot the decision boundary. 
# Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
# h = .02  # step size in the mesh
# x_delta = (x[:, 0].max() - x[:, 0].min())*0.05 # add 5% white space to border
# y_delta = (x[:, 1].max() - x[:, 1].min())*0.05
# x_min, x_max = x[:, 0].min() - x_delta, x[:, 0].max() + x_delta
# y_min, y_max = x[:, 1].min() - y_delta, x[:, 1].max() + y_delta
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = lr_predict(theta,np.c_[xx.ravel(), yy.ravel()])

# Create color maps
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

## Plot the training points
# plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)

## Show the plot
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("Logistic regression classifier")
# plt.show()