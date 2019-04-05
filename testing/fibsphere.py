import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math			
angleTheta=math.pi/6
npts=320
range_npts=int((angleTheta/math.pi)*npts)
R=4
x=[]
y=[]
z=[]
offset=2/npts
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
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.gca().set_aspect('equal', adjustable='box')
ax.scatter3D(x,y,z)
ax.set_zlim(-4,4)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()
