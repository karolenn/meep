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

x = np.linspace(-10,10,18)
y = np.cos(x)
xi = np.linspace(-10,10,202)


rbf = Rbf(x,y)
fi = rbf(xi)
print(rbf(-100))

#plt.plot(x,y,'bo')
#plt.plot(xi,fi,'g')
#plt.plot(xi,np.cos(xi),'r')
#plt.title('interpol using rbf')
#plt.show()

minimizer_kwargs= {"method": "BFGS"}
func = lambda x: rbf(x)
ret = basinhopping(func,0.5,minimizer_kwargs=minimizer_kwargs,
	niter=200)
print(ret.x,ret.fun)

####

ret2 = differential_evolution(func, [(0,10)])
print(ret2.x,ret2.fun)

###
x = np.random.rand(200)*20-10
y = np.random.rand(200)*20-10
z = np.cos(x)+np.exp(x)*np.cos(y)
ti = np.linspace(-10,10,50)
XI,YI = np.meshgrid(ti,ti)

rbf = Rbf(x,y,z)
ZI = rbf(XI,YI)

#colorplot
plt.pcolor(XI,YI,ZI,cmap=cm.jet)
plt.scatter(x,y,100,z,cmap=cm.jet)
plt.title('RBF')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.colorbar()
plt.show()

#3dplot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x,y,z,cmap=cm.jet)
ax.plot_surface(XI,YI,ZI)
plt.show()

def func3d(x):
	return(np.cos(x[0])+np.exp(x[0])*np.cos(x[1]))

def rbfmod(x):
	return(rbf(x[0],x[1]))

bounds=[(-10,10),(-10,10)]
print(len(bounds))
ret3 = differential_evolution(func3d,bounds)
ret3x = differential_evolution(rbfmod,bounds)
print(ret3.x,ret3.fun,ret3.message)
print(ret3x.x,ret3x.fun,ret3x.message)

#spos 0.08 res 60 time30
ratio=[0.06670684,0.06602146,0.05929066,0.03874125]
p_height=[2.2,3.2,4.2,4.2]
p_width=[2.2,2.2,2.2,2.0]
