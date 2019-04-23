from utils.api import *

def singleLoop(db,conf,array):
    for n in range(len(array)):
        db[str(array)].append(array[n])


def generate_simdata(db,resolution,pyramid_height,pyramid_width,source_position,polarisation, frequency):
    
    current_conf=[
    {
        "simulate": {
            "resolution": resolution,
            "use_fixed_time": False,
            "simulation_time": 30,
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
]
    write("db/sim_spec/{}.json".format(db),current_conf)
current_conf=[
    {
        "simulate": {
            "resolution": 30,
            "use_fixed_time": False,
            "simulation_time": 30,
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
]
generate_simdata("01",25,1,1,0.06,"mp.Ey",2)

db=read("db/sim_spec/01.json")
current_conf["pyramid"]["source_position"]=55