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
y = 5 - x
index = np.r_[50:100]
y[index] = 0
#Plotting
plt.figure()
plt.plot(y, 'k')
plt.xticks([50], ('c'))
plt.xlabel(r'x')
plt.ylabel(r'$L(x) = max(0,c-x)$')
plt.title(r"Plot of $L(x) = max(0, c-x)$ vs $x$")
plt.show()

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
import pandas as pd

# Configuring plot style
rc('text', usetex=True)
rc('xtick', labelsize=12) 
rc('ytick', labelsize=12) 
font = {'family' : 'serif',
        'size'   : 12}
rc('font', **font)

ql = np.genfromtxt('qlscoreLog.csv', 
                            delimiter=',')[:, 1]
ddqn = np.genfromtxt('scoreLog.csv', 
                            delimiter=',')[:, 1]
ac = np.genfromtxt('scoreLog_Actor Critic.csv', 
                            delimiter=',')[:, 1]
# Benchmark Mean Line
mean_benchmark = np.full(np.shape(steps), self.BENCHMARK_SCORE)
# Estimating Trend Line
z = np.polyfit(episodes, steps, 1)
p = np.poly1d(z)
# PLotting
plt.plot(ql, 
        'k', 
        label="Q-Learning")
plt.plot(ddqn, 
        'r', 
        label="DDQN")
plt.plot(ac, 
        'b', 
        label="Actor-Critic")
# plt.plot(episodes, 
#         mean_benchmark, 
#         'r', 
#         label="Benchmark Mean")
# Adding Legend, Title 
plt.xlabel(r'Episodes')
plt.ylabel(r'Steps')
plt.title(r"Plot of Steps vs Episode")
 # Place a legend to the right of this smaller subplot.
plt.legend(loc=4)
plt.savefig(self.PNG_PATH, bbox_inches='tight')
plt.show()

print ('Done')