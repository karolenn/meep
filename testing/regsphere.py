import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math
from copy import deepcopy
			
r=50
theta=math.pi/6
phi=math.pi*2
npts=30


def sphericalpts(r,theta,phi,npts):
	xPts = []
	yPts = []
	zPts = []
	for n in range(npts):
		angleTheta=(n/npts)*theta
		#angleTheta=angleTheta*r*math.sin(angleTheta)
		for m in range(npts):
			anglePhi=(m/npts)*phi
			#print('theta,phi',angleTheta,anglePhi)
			#print('x',r*math.sin(angleTheta)*math.cos(anglePhi))
			#print('y',r*math.sin(angleTheta)*math.sin(anglePhi))
			#print('z',-r*math.cos(angleTheta))
			xPts.append(r*math.sin(angleTheta)*math.cos(anglePhi))
			yPts.append(r*math.sin(angleTheta)*math.sin(anglePhi))
			zPts.append(-r*math.cos(angleTheta))
			#print('------')

	return xPts,yPts,zPts

def sphericalpts2(r,theta,phi,npts):
	xPts_ = []
	yPts_ = []
	zPts_ = []
	for m in range(npts):
		anglePhi=(m/(npts))*phi
		for n in range(npts):
			angleTheta=(n/npts)*theta
			#print('tet',angleTheta)
			xPts_.append(r*math.sin(angleTheta)*math.cos(anglePhi))
			yPts_.append(r*math.sin(angleTheta)*math.sin(anglePhi))
			zPts_.append(-r*math.cos(angleTheta))

	return xPts_,yPts_,zPts_

def sphericalpts3(r,theta,phi,theta_pts,phi_pts):
	xPts = []
	yPts = []
	zPts = []
	theta_angles = np.linspace(0,theta,theta_pts)
	phi_angles = np.linspace(0,phi-phi/phi_pts,phi_pts)
	print(theta_angles)
	print(phi_angles)
	for n in range(phi_pts):
		anglePhi=phi_angles[n]
		for m in range(theta_pts):
			angleTheta=theta_angles[m]
			#print('theta,phi',angleTheta,anglePhi)
			#print('x',r*math.sin(angleTheta)*math.cos(anglePhi))
			#print('y',r*math.sin(angleTheta)*math.sin(anglePhi))
			#print('z',-r*math.cos(angleTheta))
			xPts.append(r*math.sin(angleTheta)*math.cos(anglePhi))
			yPts.append(r*math.sin(angleTheta)*math.sin(anglePhi))
			zPts.append(-r*math.cos(angleTheta))
			#print('------')

	return xPts,yPts,zPts

def sphericalpts4(r,theta,phi,theta_pts,phi_pts):
	#initialize list with (0,0,r) points which we don't want to sample more than once (which happens if theta=0 and phi=(0,..,2pi))
	xPts = [0]
	yPts = [0]
	zPts = [-r]
	theta_angles = np.linspace(0+theta/theta_pts,theta,theta_pts)
	phi_angles = np.linspace(0,phi-phi/phi_pts,phi_pts)
	print(theta_angles)
	print(phi_angles)
	for m in range(theta_pts):
		angleTheta=theta_angles[m]
		for n in range(phi_pts):
			anglePhi=phi_angles[n]
			#print('theta,phi',angleTheta,anglePhi)
			#print('x',r*math.sin(angleTheta)*math.cos(anglePhi))
			#print('y',r*math.sin(angleTheta)*math.sin(anglePhi))
			#print('z',-r*math.cos(angleTheta))
			xPts.append(r*math.sin(angleTheta)*math.cos(anglePhi))
			yPts.append(r*math.sin(angleTheta)*math.sin(anglePhi))
			zPts.append(-r*math.cos(angleTheta))
			#print('------')

	return xPts,yPts,zPts
#print('x',x)
#print('y',y)
#print('z',z)
phi_pts = 6*1
theta_pts = 6
rot_points = int(phi_pts/6)
xPts,yPts,zPts = sphericalpts4(r,theta,phi,theta_pts,phi_pts)
if True:
	print('xpts',len(xPts))
	print('yPts',len(yPts))
	print('zPts',len(zPts))
	print('xpts',xPts)
	print('yPts',yPts)
	print('zPts',zPts)
fig = plt.figure()
ax = Axes3D(fig)
#plt.axis('on')
ax.scatter3D(xPts,yPts,zPts)



tot = int(phi_pts*theta_pts+1)
xPts2=deepcopy(xPts[1:])
yPts2=deepcopy(yPts[1:])
zPts2=deepcopy(zPts[1:])

for _ in range(rot_points):
	xPts2.append(xPts2.pop(0))
	yPts2.append(yPts2.pop(0))
	zPts2.append(zPts2.pop(0))
xPts2.insert(0,xPts[0])
yPts2.insert(0,yPts[0])
zPts2.insert(0,zPts[0])
print('tot',tot)
print('-')
print('lenx2',len(xPts2))
print('x2',xPts2)
print('y2',yPts2)
print('z2',zPts2)
print(xPts==xPts2)
print([0,2,1]==[1,2,0])
#xPts2,yPts2,zPts2 = sphericalpts2(r,theta,phi,npts)
#print(xPts == xPts2)
#print(len(xPts),len(xPts2))
#fig = plt.figure()
#ax = Axes3D(fig)
#plt.axis('on')
print(xPts2)
ax.scatter3D(xPts2,yPts2,zPts2,marker='v')
for n in range(tot):
	ax.text(xPts[n],yPts[n],zPts[n],n)
for n in range(tot):
	ax.text(xPts2[n],yPts2[n],zPts2[n],n,color='r',size=15)
plt.show()
