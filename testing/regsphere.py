import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math
from functions import CalculatePowerRatio 			
Theta=math.pi
Phi=math.pi*2
npts=20
r=50
x=[]
y=[]
z=[]


for n in range(npts):
	angleN=Theta*(n/npts)
		
	for m in range(npts):
		angleM=Phi*(m/npts)
		x.append(r*math.sin(angleM)*math.cos(angleN))
		y.append(r*math.sin(angleM)*math.sin(angleN))
		z.append(r*math.cos(angleM))

#print('x',x)
#print('y',y)
#print('z',z)

fig = plt.figure()
ax = Axes3D(fig)
plt.axis('on')
ax.scatter3D(x,y,z)
plt.show()
