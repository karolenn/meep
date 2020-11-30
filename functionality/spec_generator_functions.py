from .api import *
import copy
from numpy import linspace
import time
from random import uniform 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .functions import draw_uniform
from .functions import valid
from math import tan, pi




###Some functionality to generate pyramid specifications

###Generate equally distanced sample points
def generate_eq_dist(x, y, z):
    tests = []
    for itx in linspace(x["from"], x["to"], x["steps"]):
        for ity in linspace(y["from"], y["to"], y["steps"]):
            for itz in linspace(z["from"], z["to"], z["steps"]):
                tests.append((itx,ity,itz))
    if True:
        ty = (y["to"]-y["from"])/2+y["from"]
        tx = (x["to"]-x["from"])/2+x["from"]
        tz = (z["to"]-z["from"])/2+z["from"]
        tests.append((tx,ty,tz))
    return tests

###Generate equally distanced sampled points with varying pyramid width and source position. Pyramid tip and base angle is 62 degrees.
def generate_eq_dist_fixed_angle(y, z):
    tests = []
    pyr_height = tan((pi*62)/180)/2
    for ity in linspace(y["from"], y["to"], y["steps"]):
        itx = ity*pyr_height
        for itz in linspace(z["from"], z["to"], z["steps"]):
            tests.append((itx,ity,itz))
    if True: #Generate center point for RBF as well
        ty = (y["to"]-y["from"])/2+y["from"]
        tx = ty*pyr_height
        tz = (z["to"]-z["from"])/2+z["from"]
        tests.append((tx,ty,tz))
    return tests


def generate_rand_dist(limit_x, limit_y, limit_z, radius, satisfied, max_time = 3):
	#check if we already have some data points from simulations
	selected=[]
	for _ , x in limit_x.items():
		for _ , y in limit_y.items():
			for _ , z in limit_z.items():
				if (x,y,z) not in selected:
					selected.append((x,y,z))

	t = time.time()

	while time.time() - t < max_time:
		if len(selected) >= satisfied:
		#	print('selected',selected)
			break
		x, y, z = draw_uniform(limit_x, limit_y, limit_z)
		if valid(x,y,z, selected, radius):
			selected.append((round(x,2), round(y,2), round(z,2)))
	return selected


###Convert generate point to json entry
def points_to_json(points, template):
    tests = []
    for x,y,z in points:
        tmp = copy.deepcopy(template)
        tmp["pyramid"]["pyramid_height"] = x
        tmp["pyramid"]["pyramid_width"] = y
        tmp["pyramid"]["source_position"] = z
        tests.append(tmp)
    return tests
