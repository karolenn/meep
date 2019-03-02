import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import math as math
import time
#from functions import CalculatePowerRatio 			#import function that calculates ratio of total and angle out


def fibspherepts(r,theta,npts):
	startTime=time.time()
	x=[]
	y=[]
	z=[]

	offset=2/npts
	range_npts=int((theta/math.pi)*npts)
	increment = math.pi*(3 - math.sqrt(5))

	for n in range(npts):
		y.append(r*((n*offset-1)+(offset/2)))
		R = r*math.sqrt(1-pow(y/r,2))
		phi = (n % npts)*increment
		x.append(R*math.cos(phi))
		z.append(R*math.sin(phi))
	
	print('time taken:',time.time()-startTime)
	
	
