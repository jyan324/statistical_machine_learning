import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

#Setting Matplotlib to display correctly
rc('text', usetex=True)
rc('xtick', labelsize=20) 
rc('ytick', labelsize=20) 
font = {'family' : 'serif',
        'size'   : 22}
rc('font', **font)

# Specifying the x axis
x = np.linspace(0.0, 10.0, num=100)
y = np.zeros(100)
index = np.r_[39:59]
y[index] = 1
#Plotting
plt.figure()
plt.plot(y, 'k')
plt.xticks([39, 59], ('a', '(a+1)'))
plt.xlabel(r'x')
plt.ylabel(r'$p_{X|Y}(x|1)$')
plt.title(r"Plot of $p_{X|Y}(x|1)$ vs $x$")
plt.show()