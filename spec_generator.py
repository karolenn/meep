from functionality.api import *
import copy
from numpy import linspace
import time
from random import uniform 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functionality.spec_generator_functions import *
from math import tan, pi

###spec_generator creates a .json file in db/sim_spec that is a template for initial_runs & main.py to run.

###Name of the sim_spec.json file
sim_spec_filename = 'qw60_011b'


###We now have 3 ways to create different pyramids to be simulated. 'eq dist', 'eq dist with fixed angle between base and top' and 'rand' way
###x corresponds to pyramid height, y to pyramid width, z to source position for all three functions below
#
def equaldistance(template,sim_spec_filename):
    x = {"from": 0.6, "to":1 ,"steps": 2}
    y = {"from": 0.6, "to":1 ,"steps": 2}
    z = {"from": 0.01, "to": 0.9 ,"steps": 2}
    result = generate_eq_dist(x, y, z)
    tests = points_to_json(result, template)
    
    x,y,z = zip(*result)
    ax = plt.axes(projection='3d')
    plt.title('Pyramids to be simulated')
    ax.set_xlabel("pyramid height")
    ax.set_ylabel("pyramid width")
    ax.set_zlabel("source position")
    ax.scatter(x,y,z)
    plt.show()
    write("db/sim_spec/{}.json".format(sim_spec_filename), tests)

###Generate equally distance points with fixed base-tip angle relationship. Center point in search space is included.
def fixedangle(template,sim_spec_filename):
    y = {"from": 0.5, "to":0.8, "steps": 2} #pyramid width
    z = {"from": 0.1, "to": 0.4, "steps": 2} #source position
    result =  generate_eq_dist_fixed_angle(y,z) 
    tests = points_to_json(result, template)
    
    x,y,z = zip(*result)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title('Pyramids to be simulated')
    ax.set_xlabel("pyramid width")
    ax.set_ylabel("pyramid source position")
    ax.scatter(y,z)
    plt.show()
    write("db/sim_spec/{}.json".format(sim_spec_filename), tests)


###Generate randomly sampled initial points
def minmaxrand(template,sim_spec_filename):
    
    x = {"from": 0.4, "to":0.8 }
    y = {"from": 0.4, "to":0.8 }
    z = {"from": 0.1, "to": 0.6 }
    result = generate_rand_dist(x, y, z, 0.5, 16)
    print(result)
    x,y,z = zip(*result)

    tests = points_to_json(result, template)

    ax = plt.axes(projection='3d')
    plt.title('Pyramids to be simulated')
    ax.set_xlabel("pyramid height")
    ax.set_ylabel("pyramid width")
    ax.set_zlabel("source position")
    ax.scatter(x,y,z)
    plt.show()
    write("db/sim_spec/{}.json".format(sim_spec_filename), tests)


def generate_qw(template,sim_spec_filename):
    nr_of_dipoles = 90
    polarization = [(1,0,0),(0,1,0),(0,0,1)]
    pyramid_list = []
    for j in range(nr_of_dipoles):
        source_pos = (0,uniform(-1,1),uniform(0,1))
        #source_pos = (0,0,0)
        for k in range(len(polarization)):
            tmp = copy.deepcopy(template)
            tmp["pyramid"]["source_position"] = source_pos
            tmp["pyramid"]["source_direction"] = polarization[k]
            pyramid_list.append(tmp)

    write("db/sim_spec/{}.json".format(sim_spec_filename), pyramid_list)

###This is where you choose how to sample your initial runs using one of the three functions above, i.e equaldistance OR fixedangle OR minmaxrand
if __name__ == "__main__":


    template={
        "simulate": {
            "resolution": 60,
            "use_fixed_time": False,
            "simulation_time": 60,
            "dpml": 0.1,
            "padding": 0.025,
            "ff_pts": 30,
            "ff_calc": "Below",
            "ff_cover": False,
            "use_symmetries": False,
            "calculate_flux": True,
            "ff_calculations": True,
            "ff_angle": 6,
            "simulation_ratio": "6/5",
            "substrate_ratio": "1/10",
            "quantum_well": True,
            "polarization_in_plane": False
        },
        "pyramid": {
            "source_position": (0,0.2,0.3451462262234885),
            "pyramid_height": 0.814,
            "pyramid_width": 1,
            "truncation": 0,
            "source_direction": (0,0,0),
            "frequency_center": 2,
            "frequency_width": 0.8,
            "number_of_freqs": 5,
            "cutoff": 4
        },
        "result": {}
    }

    generate_qw(template,sim_spec_filename)


##This is the template file for generating a new simulation specification db. 








