import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rc
from sklearn import neighbors, datasets
from tqdm import tqdm

#Setting Matplotlib to display correctly
rc('xtick', labelsize=20) 
rc('ytick', labelsize=20) 
font = {'family' : 'serif',
        'size'   : 22}
rc('font', **font)

np.random.seed(2017) # Set random seed so results are repeatable

n = 5000 # number of training points
k = 75 # number of neighbors to consider 
scores = [] # list containing score corresponding to different values of k

## Generate a simple 2D dataset
X, y = datasets.make_moons(n,'True',0.3)

# Comment if you want to plot Optimization Curve
classifier = neighbors.KNeighborsClassifier(k,'uniform')
classifier.fit(X, y)
print ("\t kNN Performance Report\n")
print ("n = %i, Max Score = %f, k = %i\n" %(n, classifier.score(X, y), k))

#############################################################################
#Uncomment for Not Plotting Optimization Curve
# domain = range(3,n+1, 2)
# # Uncomment if n > 2500
# # domain1 = [i for i in range(3, 500, 2)] 
# # domain2 = [i for i in range(501, 1000, 50)]
# # domain3 = [i for i in range(1001,5001, 250)]
# # domain = domain1+domain2+domain3
# ## Checking k for all values
# for k in tqdm(domain):
#     ## Create instance of KNN classifier
#     classifier = neighbors.KNeighborsClassifier(k,'uniform')
#     classifier.fit(X, y)
#     scores.append(classifier.score(X,y))

# ## Plot the score graph and analyze
# scores = np.array(scores)
# max_score = scores.max() # get the largest value of score
# #optimal_k = 2*np.argmax(np.array(scores)) + 3 # get the optimal k value
# optimal_k = domain[np.argmax(np.array(scores))]
# print (optimal_k)
# print ("\t kNN Optimization Report\n")
# print ("n = %i, Max Score = %f, k = %i\n" %(n, max_score, optimal_k))
# plt.plot(domain , scores, 'k')
# plt.title("kNN Optimization curve for n = %i datapoints" %(n))
# plt.ylabel("Score")
# plt.xlabel("k")
# plt.ylim (0, 1)
# plt.grid(True, color='k', linestyle='--', linewidth=1)
# plt.show()

# ## Optimal
# classifier = neighbors.KNeighborsClassifier(optimal_k,'uniform')
# classifier.fit(X, y)
#############################################################################

## Plot the decision boundary. 
# Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
h = .02  # step size in the mesh
x_delta = (X[:, 0].max() - X[:, 0].min())*0.05 # add 5% white space to border
y_delta = (X[:, 1].max() - X[:, 1].min())*0.05
x_min, x_max = X[:, 0].min() - x_delta, X[:, 0].max() + x_delta
y_min, y_max = X[:, 1].min() - y_delta, X[:, 1].max() + y_delta
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])


# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

## Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-NN classifier trained on %i data points" % (k,n))

## Show the plot
plt.show()