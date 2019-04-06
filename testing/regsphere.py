import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math
			
r=10
theta=math.pi
phi=math.pi*2
npts=40

xPts=[]
yPts=[]
zPts=[]


def sphericalpts(r,theta,phi,npts,xPts,yPts,zPts):

	for n in range(npts):
		angleTheta=(n/npts)*theta
		for m in range(npts):
			anglePhi=(m/npts)*phi
			xPts.append(r*math.sin(angleTheta)*math.cos(anglePhi))
			yPts.append(r*math.sin(angleTheta)*math.sin(anglePhi))
			zPts.append(-r*math.cos(angleTheta))

	return(xPts,yPts,zPts)

def sphericalpts2(r,theta,phi,npts,xPts,yPts,zPts):

	for m in range(npts):
		anglePhi=(m/npts)*phi
		for n in range(npts):
			angleTheta=(n/npts)*theta
			xPts.append(r*math.sin(angleTheta)*math.cos(anglePhi))
			yPts.append(r*math.sin(angleTheta)*math.sin(anglePhi))
			zPts.append(-r*math.cos(angleTheta))

	return(xPts,yPts,zPts)

#print('x',x)
#print('y',y)
#print('z',z)
sphericalpts(r,theta,phi,npts,xPts,yPts,zPts)

fig = plt.figure()
ax = Axes3D(fig)
plt.axis('on')
ax.scatter3D(xPts,yPts,zPts)
plt.show()

xPts=[]
yPts=[]
zPts=[]

sphericalpts2(r,theta,phi,npts,xPts,yPts,zPts)
fig = plt.figure()
ax = Axes3D(fig)
plt.axis('on')
ax.scatter3D(xPts,yPts,zPts)
plt.show()
