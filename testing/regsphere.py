import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math
from copy import deepcopy
			
r=50
theta=math.pi/2-math.pi/100
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
	#the value of angles to loop through
	theta_angles = np.linspace(0+theta/theta_pts,theta,theta_pts)
	phi_angles = np.linspace(0,phi-phi/phi_pts,phi_pts)
	print(theta_angles)
	print(phi_angles)
	for m in range(theta_pts):
		angleTheta=theta_angles[m]
		for n in range(phi_pts):
			anglePhi=phi_angles[n]
			xPts.append(r*math.sin(angleTheta)*math.cos(anglePhi))
			yPts.append(r*math.sin(angleTheta)*math.sin(anglePhi))
			zPts.append(-r*math.cos(angleTheta))
	return xPts,yPts,zPts
#print('x',x)
#print('y',y)
#print('z',z)
phi_pts = 12
theta_pts = 3
rotation_int = 1
rot_points = int(rotation_int*phi_pts/6)
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
#we need to split up [x1,..,xn] to [x1,..,xi],[xi+1,xj] based on number of phi pts per theta
#split x,y,z, into corresponding theta
x_tmp = [0]*(tot-1)
y_tmp = [0]*(tot-1)
z_tmp = [0]*(tot-1)
for i in range(0,len(xPts2),phi_pts):
	print('i',i)
	xPtsRing = deepcopy(xPts2[i:i+phi_pts])
	yPtsRing = deepcopy(yPts2[i:i+phi_pts])
	zPtsRing = deepcopy(zPts2[i:i+phi_pts])
	for _ in range(rot_points):
		xPtsRing.append(xPtsRing.pop(0))
		yPtsRing.append(yPtsRing.pop(0))
		zPtsRing.append(zPtsRing.pop(0))
	x_tmp[i:i+phi_pts] = xPtsRing
	y_tmp[i:i+phi_pts] = yPtsRing
	z_tmp[i:i+phi_pts] = zPtsRing
x_tmp.insert(0,xPts[0])
y_tmp.insert(0,yPts[0])
z_tmp.insert(0,zPts[0])
print('xtmp',x_tmp)
for _ in range(rot_points):
	xPts2.append(xPts2.pop(0))
	yPts2.append(yPts2.pop(0))
	zPts2.append(zPts2.pop(0))
xPts2.insert(0,xPts[0])
yPts2.insert(0,yPts[0])
zPts2.insert(0,zPts[0])
if True:
	print('tot',tot)
	print('-')
	print('lenx2',len(x_tmp))
	print('x2',xPts2)
	#print('y2',yPts2)
	#print('z2',zPts2)
	print(xPts==xPts2)

#xPts2,yPts2,zPts2 = sphericalpts2(r,theta,phi,npts)
#print(xPts == xPts2)
#print(len(xPts),len(xPts2))
#fig = plt.figure()
#ax = Axes3D(fig)
plt.axis('on')

ax.scatter3D(x_tmp,y_tmp,z_tmp,marker='v',color='orange')
for n in range(tot):
	ax.text(xPts[n],yPts[n],zPts[n],n)
for n in range(tot):
	ax.text(x_tmp[n],y_tmp[n],z_tmp[n]+5,n,color='r',size=10)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()
