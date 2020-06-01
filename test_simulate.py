from structure.pyramid import Pyramid
import sys
import datetime

#FOR RUNNING ONE SIMULATION WITH THE CONFIG BELOW

config = {
    "simulate": {
        "resolution": 60,
        "use_fixed_time": False,
        "simulation_time": 10,
        "dpml": 0.1,
        "padding": 0.025,
        "ff_pts": 400,
        "ff_calc": "Above",
        "ff_cover": False,
        "use_symmetries": False,
        "calculate_flux": True,
        "ff_calculations": True,
        "ff_angle": 6,
        "simulation_ratio": "6/5",
        "substrate_ratio": "1/10",
        "quantum_well": True
    },
    "pyramid": {
        "source_position": (0,0.2,0.3451462262234885),
        "pyramid_height": 1.0835834397989268,
        "pyramid_width": 1.1523030698665553,
        "truncation": 0,
        "source_direction": "mp.Ey",
        "frequency_center": 2,
        "frequency_width": 1.2,
        "number_of_freqs": 3,
        "cutoff": 4
        },
        "result": {}
    }  

if (len(sys.argv)) != 1:
		print("Not enough arguments")
		exit(0)

    
pyramid = Pyramid()
pyramid.setup(config["pyramid"])
result = pyramid.simulate(config["simulate"])
print(config)
print(result)
print('Simulation finished at:',datetime.datetime.now())
