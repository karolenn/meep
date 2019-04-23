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

ph=[2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
ff=[0.04303790391296904, 0.04446679668929159, 0.04716459353793865, 0.05119275907961527, 0.05575515108079563, 0.05875465088170854, 0.06320563361553232, 0.06911011743519525, 0.07346700508779018, 0.07980325351819721, 0.08595140544050793, 0.09164674662008034, 0.0994513711736578, 0.10720208557788029, 0.11258742874958047, 0.11839849443347393, 0.12537682285139037, 0.13027205235350844, 0.1358709711219651, 0.13917108256916358, 0.14193883431461266]
xi=np.linspace(min(ph),max(ph))

rbf = Rbf(ph,ff)
fi = rbf(xi)
print(rbf(xi))

#plt.plot(ph,ff,'bo')
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
X=[]
Y=[]
num=3
for i in range(num):
	X.append(1+i)
	for j in range(num):
		Y.append(1+j)
print(X,Y)

x=[1,1,1,2,2,2,3,3,3]
y=[1,2,3,1,2,3,1,2,3]
z=[10,8,7,8,6,7,7,6,5]
RBF_Func=Rbf(x,y,z)
xi = np.linspace(min(x),max(x))
yi = np.linspace(min(y),max(y))
zi = RBF_Func(xi,yi)
fig = plt.figure()
ax = plt.axes(projection='3d')
c = [n*10+3 for n in range(len(x))]
print(c)
ax.scatter(x,y,z,c,color='g',alpha=1)

XI,YI=np.meshgrid(xi,yi)
ZI = RBF_Func(XI,YI)
ax.plot_surface(XI,YI,ZI,cmap=cm.jet)
plt.show()
print(RBF_Func(2,2))

#x = np.random.rand(200)*20-10
#y = np.random.rand(200)*20-10
#z = np.cos(x)+np.exp(x)*np.cos(y)
#ti = np.linspace(-10,10,50)
#XI,YI = np.meshgrid(ti,ti)

#rbf = Rbf(x,y,z)
#ZI = rbf(XI,YI)

#colorplot
#plt.pcolor(XI,YI,ZI,cmap=cm.jet)
#plt.scatter(x,y,100,z,cmap=cm.jet)
#plt.title('RBF')
#plt.xlim(-10,10)
#plt.ylim(-10,10)
#plt.colorbar()
#plt.show()

#3dplot
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter(x,y,z,cmap=cm.jet)
#ax.plot_surface(XI,YI,ZI)
#plt.show()

def func3d(x):
	return(np.cos(x[0])+np.exp(x[0])*np.cos(x[1]))

def rbfmod(x):
	return(rbf(x[0],x[1]))

#bounds=[(-10,10),(-10,10)]
#print(len(bounds))
#ret3 = differential_evolution(func3d,bounds)
#ret3x = differential_evolution(rbfmod,bounds)
#print(ret3.x,ret3.fun,ret3.message)
#print(ret3x.x,ret3x.fun,ret3x.message)

#spos 0.08 res 60 time30
ratio=[0.06670684,0.06602146,0.05929066,0.03874125]
p_height=[2.2,3.2,4.2,4.2]
p_width=[2.2,2.2,2.2,2.0]
