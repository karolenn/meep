from lib.pyramid import Pyramid
import sys

#FOR RUNNING ONE SIMULATION WITH THE CONFIG BELOW

config = {
        "simulate": {
            "resolution": 60,
            "use_fixed_time": False,
            "simulation_time": 10,
            "dpml": 0.1,
            "padding": 0.025,
            "ff_pts": 1600,
            "ff_calc": "Both",
            "ff_cover": True,
            "use_symmetries": True,
            "calculate_flux": True,
            "ff_calculations": True,
            "ff_angle": 1,
            "simulation_ratio": "6/5",
            "substrate_ratio": "1/10"
        },
        "pyramid": {
            "source_position": 0.01,
            "pyramid_height": 0.5,
            "pyramid_width": 0.5,
            "truncation": 0,
            "source_direction": "mp.Ez",
            "frequency_center": 2,
            "frequency_width": 1,
            "number_of_freqs": 10,
            "cutoff": 2
        },
        "result": {}
    }  

if (len(sys.argv)) != 1:
		print("Not enough arguments")
		exit(0)

    
pyramid = Pyramid()
pyramid.setup(config["pyramid"])
result = pyramid.simulate(config["simulate"])
print(result)
