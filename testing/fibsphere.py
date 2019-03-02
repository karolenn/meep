import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math			
angleTheta=math.pi
npts=400
range_npts=int((angleTheta/math.pi)*npts)
R=10
x=[]
y=[]
z=[]
offset=2/npts
increment = math.pi*(3 - math.sqrt(5))

for n in range(range_npts):
	Y=R*((n*offset-1)+(offset/2))
	r = R*math.sqrt(1-pow(Y/R,2))
	phi = (n % npts)*increment
	y.append(Y)
	x.append(r*math.cos(phi))
	z.append(r*math.sin(phi))



print('x',x)
print('y',y)
print('z',z)

fig = plt.figure()
ax = Axes3D(fig)
#plt.xlim(-3,3)
plt.ylim(-10,10)
plt.gca().set_aspect('equal', adjustable='box')
ax.scatter3D(x,y,z)
plt.show()
