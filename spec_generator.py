from utils.api import *
import copy
from numpy import linspace
import time
from random import uniform 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def valid(x,y,z, selected, radius):
    dis = lambda f,s: (f-s)**2
    #print(selected)
    for ptx, pty,ptz in selected:
        if (dis(ptx,x)+ dis(pty,y)+dis(ptz,z))**(1/2) < radius:
            return False
    return True


def get_next(limit_x, limit_y, limit_z):
    x = uniform(limit_x["from"], limit_x["to"])
    y = uniform(limit_y["from"], limit_y["to"])
    z = uniform(limit_z["from"], limit_z["to"])
    return x, y, z



def generate_rand_dist(limit_x, limit_y, limit_z, radius, satisfied, max_time = 3):
    selected = []
    for _ , x in limit_x.items():
        for _ , y in limit_y.items():
            for _ , z in limit_z.items():
                if (x,y,z) not in selected:
                    selected.append((x,y,z))
    #print(selected)
    t = time.time()
    while time.time() - t < max_time:
        if len(selected) >= satisfied:
            print(selected)
            break
        x, y, z = get_next(limit_x, limit_y, limit_z)
        if valid(x,y,z, selected, radius):
            selected.append((x, y, z))
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


def test(template):
    x = {"from": 1, "to":2 ,"steps": 5}
    y = {"from": 2, "to":4 ,"steps": 4}
    z = {"from": 0.01, "to": 0.5 ,"steps": 4}
    result = generate_eq_dist(x, y, z)
    tests = points_to_json(result, template)
    
    x,y,z = zip(*result)
    ax = plt.axes(projection='3d')
    ax.scatter(x,y,z)
    plt.show()
    write("db/sim_spec/test.json", tests)
    #print(tests)

def testRand(template):
    
    x = {"from": 1, "to":3 }
    y = {"from": 1, "to":3 }
    z = {"from": 0.06, "to": 0.06 }
    result = generate_rand_dist(x, y, z, 0.5, 9)
    x,y,z = zip(*result)

    tests = points_to_json(result, template)

    ax = plt.axes(projection='3d')
    ax.scatter(x,y,z)
    plt.show()
    write("db/sim_spec/test.json", tests)
    #print(tests)


if __name__ == "__main__":
    template={
        "simulate": {
            "resolution": 20,
            "use_fixed_time": True,
            "simulation_time": 1,
            "dpml": 0.1,
            "padding": 0.1,
            "ff_pts": 1600,
            "ff_cover": False,
            "use_symmetries": True,
            "calculate_flux": True,
            "ff_calculations": True,
            "ff_angle": 6,
            "simulation_ratio": "6/5",
            "substrate_ratio": "1/20"
        },
        "pyramid": {
            "source_position": 0.06,
            "pyramid_height": 3.1,
            "pyramid_width": 2,
            "source_direction": "mp.Ey",
            "frequency_center": 2,
            "frequency_width": 0.5,
            "number_of_freqs": 1,
            "cutoff": 2
        },
        "result": {}
    }  
    test(template)