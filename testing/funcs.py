import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from mpl_toolkits.mplot3d import Axes3D
import math as math
from scipy.interpolate import Rbf
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution

t=np.linspace(0,140,606)
def f(t):
	return(int(60/(t+1)))
fval=[]
print(len(t))
for i in range(len(t)):
	fval.append(f(t[i]))


plt.plot(t,fval)
plt.show()
print(round(2.3))
