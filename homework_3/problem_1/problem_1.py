from sklearn.datasets import load_boston
import numpy as np
boston = load_boston()
x = boston.data
y = boston.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    train_size=400,
                                                    random_state=13)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_std = scaler.transform(x_train)

X = np.concatenate([np.ones((x_train_std.shape[0], 1)), x_train_std], axis=1).astype(np.float)
X_test = np.concatenate([np.ones((x_test.shape[0], 1)), scaler.transform(x_test)], axis=1).astype(np.float)


theta_opt = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), 
                      y_train.astype(np.float))

from sklearn.metrics import mean_squared_error
pred = np.matmul(X_test, theta_opt)
error = mean_squared_error(y_test, pred)
print ("MSE for Least Square Model = %.4f"%(error))

def Gamma(d, Lambda):
    """
        Function to generate Gamma
    """
    I = np.identity(d)
    O_00 = 0
    O_01 = np.zeros((1,d))
    O_10 = np.zeros((d,1))
    gamma = np.block([[O_00, O_01], 
                       [O_10, Lambda*I]]);
    return gamma

def ridgeRegression(X,Y, Lambda):
    """
        Function to find optimal weights for ridge regression
    """
    d = X.shape[1]-1
    gamma = Gamma(d, np.sqrt(Lambda))
    X_inv = np.linalg.inv(np.matmul(np.transpose(X), X) + np.matmul(np.transpose(gamma), gamma))
    theta_opt = np.matmul(np.matmul(X_inv, np.transpose(X)), Y)
    return theta_opt

def evaluateRidgeRegression(X,Y,theta):
    """
        Function to evaluate performance of ridge Regression using MSE
    """
    pred = np.matmul(X, theta)
    error = mean_squared_error(Y, pred)
    return error

X_train, X_val, Y_train, Y_val = train_test_split(X, 
                                                    y_train, 
                                                    test_size=0.30,
                                                    random_state=13)

errors = []
Lambdas = np.arange(0.1,100,0.1).tolist()
for Lambda in Lambdas:
    theta = ridgeRegression(X_train, Y_train, Lambda)
    error = evaluateRidgeRegression(X_val, Y_val, theta)
    errors.append(error)

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=20) 
rc('ytick', labelsize=20) 
font = {'family' : 'serif',
        'size'   : 22}
rc('font', **font)

plt.figure()
plt.plot(Lambdas, errors, 'k')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'Mean Square Error')
plt.title(r"Plot of MSE vs $\lambda$")
plt.show()

Lambda_opt = Lambdas[errors.index(min(errors))]
print ("The optimal value of lambda is %.4f"%Lambda_opt)
print ("The value of MSE corresponding to minimum Lambda is %.4f"%min(errors))
theta_opt = ridgeRegression(X_train, Y_train, Lambda_opt)
error = evaluateRidgeRegression(X_test, y_test, theta_opt)
print ("MSE for Ridge Regression Model = %.4f"%(error))


X_train, X_val, Y_train, Y_val = train_test_split(x_train_std, 
                                                    y_train, 
                                                    test_size=0.3,
                                                    random_state=13)

from sklearn import linear_model
errors = []
alphas = np.arange(0.1,10,0.1).tolist()
for alpha in alphas:
    reg = linear_model.Lasso(alpha)
    reg.fit(X_train,Y_train)
    pred = reg.predict(X_val)
    error = mean_squared_error(Y_val, pred)
    errors.append(error)
# Plotting the error
plt.figure()
plt.plot(alphas, errors, 'k')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'Mean Square Error')
plt.title(r"Plot of MSE vs $\alpha$")
plt.show()
# Finding the Optimal alpha
alpha_opt = alphas[errors.index(min(errors))]
print ("The optimal value of alpha is %.4f"%alpha_opt)
print ("The value of MSE corresponding to minimum alpha is %.4f"%min(errors))
# Finding Optimal Model
reg_opt = linear_model.Lasso(alpha_opt)
reg_opt.fit(X_train,Y_train)

pred = reg_opt.predict(scaler.transform(x_test))
error = mean_squared_error(y_test, pred)
print ("MSE for LASSO = %.4f"%(error))

