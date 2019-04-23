from utils.api import *
import copy
from numpy import linspace

def singleLoop(db,conf,array):
    for n in range(len(array)):
        db[str(array)].append(array[n])

def generate_eq_dist(x, y, z, template):
    tests = []
    for itx in linspace(x["from"], x["to"], x["steps"]):
        for ity in linspace(y["from"], y["to"], y["steps"]):
            for itz in linspace(z["from"], z["to"], z["steps"]):
                tmp = copy.deepcopy(template)
                tmp["pyramid"][z["name"]] = itz
                tmp["pyramid"][x["name"]] = itx
                tmp["pyramid"][y["name"]] = ity
                tests.append(tmp)
    return tests

def test():
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
    x = {"name": "pyramid_height", "from": 1, "to":2 ,"steps": 5}
    y = {"name": "pyramid_width", "from": 2, "to":4 ,"steps": 4}
    z = {"name": "source_position", "from": 0.01, "to": 0.5 ,"steps": 4}
    result = generate_eq_dist(x, y, z, template)
    write("db/sim_spec/out.json", result)
    #print(result)

if __name__ == "__main__":  
    test()