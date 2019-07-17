from utils.api import *
import copy
from numpy import linspace
import time
from random import uniform 
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from utils.functions import get_next
from utils.functions import valid
from math import tan, pi

def singleLoop(db,conf,array):
    for n in range(len(array)):
        db[str(array)].append(array[n])

def generate_eq_dist(x, y, z):
    tests = []
    for itx in linspace(x["from"], x["to"], x["steps"]):
        for ity in linspace(y["from"], y["to"], y["steps"]):
            for itz in linspace(z["from"], z["to"], z["steps"]):
                tests.append((itx,ity,itz))
    return tests

def generate_eq_dist_fixed_angle(y, z):
    tests = []
    pyr_height = tan((pi*62)/180)/2
  #  for itx in linspace(x["from"], x["to"], x["steps"]):
    for ity in linspace(y["from"], y["to"], y["steps"]):
        itx = ity*pyr_height
        for itz in linspace(z["from"], z["to"], z["steps"]):
            tests.append((itx,ity,itz))
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
		x, y, z = get_next(limit_x, limit_y, limit_z)
		if valid(x,y,z, selected, radius):
			selected.append((round(x,2), round(y,2), round(z,2)))
	return selected


def points_to_json(points, template):
    tests = []
    for x,y,z in points:
        tmp = copy.deepcopy(template)
        tmp["pyramid"]["pyramid_height"] = x
        tmp["pyramid"]["pyramid_width"] = y
        tmp["pyramid"]["source_position"] = z
        tests.append(tmp)
    return tests


#x corresponds to pyramid height, y to pyramid width, z to source position for all three functions below
#
def test(template):
    x = {"from": 1, "to":1 ,"steps": 1}
    y = {"from": 1, "to":1 ,"steps": 1}
    z = {"from": 0.01, "to": 0.4 ,"steps": 1}
    result = generate_eq_dist(x, y, z)
    tests = points_to_json(result, template)
    
    x,y,z = zip(*result)
 #   ax = plt.axes(projection='3d')
  #  ax.scatter(x,y,z)
   # plt.show()
    write("db/sim_spec/analytic2.json", tests)
    #print(tests)

def fixedangle(template):
    y = {"from": 1, "to":8, "steps": 10} #pyramid width
    z = {"from": 0.01, "to": 0.9, "steps": 10} #source position
   # pyr_height = tan((pi*62)/180)/2
   # x = {"from": y["from"]*pyr_height,"to": y["to"]*pyr_height, "steps": y["steps"]}
    result =  generate_eq_dist_fixed_angle(y,z) 
    tests = points_to_json(result, template)
    
    x,y,z = zip(*result)
    #ax = plt.axes(projection='3d')
    #ax.set_xlabel("pyramid height")
    #ax.set_ylabel("pyramid width")
    #ax.scatter(x,y,z)
    #plt.show()
    write("db/sim_spec/f1.json", tests)

def testRand(template):
    
    x = {"from": 0.8, "to":0.8 }
    y = {"from": 0.6, "to":1.2 }
    z = {"from": 0.1, "to": 0.6 }
    result = generate_rand_dist(x, y, z, 0.5, 4)
    x,y,z = zip(*result)

    tests = points_to_json(result, template)

    #ax = plt.axes(projection='3d')
    #ax.scatter(x,y,z)
    #plt.show()
    write("db/sim_spec/laptop.json", tests)
    #print(tests)


if __name__ == "__main__":
    template={
        "simulate": {
            "resolution": 60,
            "use_fixed_time": False,
            "simulation_time": 10,
            "dpml": 0.1,
            "padding": 0.025,
            "ff_pts": 1600,
            "ff_calc": "Both",
            "ff_cover": False,
            "use_symmetries": True,
            "calculate_flux": True,
            "ff_calculations": True,
            "ff_angle": 6,
            "simulation_ratio": "6/5",
            "substrate_ratio": "1/10"
        },
        "pyramid": {
            "source_position": 0.2,
            "pyramid_height": 0.5,
            "pyramid_width": 0.5,
            "truncation": 0,
            "source_direction": "mp.Ey",
            "frequency_center": 2,
            "frequency_width": 1.2,
            "number_of_freqs": 6,
            "cutoff": 4
        },
        "result": {}
    }  
    fixedangle(template)
