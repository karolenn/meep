import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math			
angleTheta=math.pi/6
npts=1800
range_npts=int((angleTheta/math.pi)*npts)
R=47
x=[]
y=[]
z=[]
offset=0.8/npts
increment = math.pi*(3 - math.sqrt(5))

for n in range(range_npts):
	Z=R*((n*offset-1)+(offset/2))
	r = R*math.sqrt(1-pow(Z/R,2))
	phi = (n % npts)*increment
	z.append(Z)
	x.append(r*math.cos(phi))
	y.append(r*math.sin(phi))



fig = plt.figure()
ax = Axes3D(fig)
#plt.xlim(-4,4)
#plt.ylim(-4,4)
#plt.gca().set_aspect('equal', adjustable='box')
ax.scatter3D(x,y,z)
#for n in range(range_npts):
	#ax.text(x[n],y[n],z[n],n)
ax.set_zlim(-100,100)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
print('range npts',range_npts)
print('x',x)
print('y',y)

#plt.show()

#rotate field
x_r = []
y_r = []
n=5
for i in range(range_npts):
	x_r.append(x[i]*math.cos(n*math.pi/3)-math.sin(n*math.pi/3)*y[i])
	y_r.append(x[i]*math.sin(n*math.pi/3)+math.cos(n*math.pi/3)*y[i])
print('xr',x_r)
print('yr',y_r)
ax.scatter3D(x_r,y_r,z)
#for n in range(range_npts):
#	ax.text(x_r[n],y_r[n],z[n],n)
plt.show()