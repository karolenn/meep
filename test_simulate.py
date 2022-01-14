from structure.pyramid import Pyramid
from structure.silverball import Balls
import sys
import datetime
from functionality.api import polar_to_complex_conv,sim_to_json,write_result, complex_to_polar_conv

#FOR RUNNING ONE SIMULATION WITH THE CONFIG BELOW

simulation = "pyramid"

config_pyramid = {
        "simulate": {
            "resolution":100,
            "use_fixed_time": False,
            "simulation_time": 2,
            "dpml": 0.1,
            "padding": 0.025,
            "ff_pts": 800,
            "ff_calc": "Below",
            "ff_cover": False,
            "use_symmetries": True,
            "calculate_flux": True,
            "calculate_source_flux": True,
            "source_flux_pixel_size": 8,
            "ff_calculations": True,
            "ff_angle": 6,
            "fibb_sampling": True,
            "simulation_ratio": "7/5",
            "substrate_ratio": "2/10",
            "output_ff": False,
            "geometry": False,
            "material_function": "truncPyramid",
            "polarization_in_plane": False
        },
        "pyramid": {
            "source_position": [0,0,0.25],
            "pyramid_height": 0.94,
            "pyramid_width": 1,
            "truncation_width": 0.1,
            "CL_thickness": 0.1,
            "CL_material":"Ag",
            "source_direction": [1,0,0],
            "source_on_wall": False,
            "frequency_center": 1.85,
            "frequency_width": 0.76,
            "number_of_freqs": 3,
            "cutoff": 4
        },
        "result": {}
    }  

config_balls = {
       "simulate": {
            "resolution": 750,
            "use_fixed_time": True,
            "simulation_time": 60,
            "dpml": 0.01,
            "padding": 0.0025,
            "ff_pts": 800,
            "ff_calc": "Below",
            "ff_cover": False,
            "use_symmetries": True,
            "calculate_flux": True,
            "calculate_source_flux": True,
            "source_flux_pixel_size": 8,
            "ff_calculations": False,
            "ff_angle": 6,
            "fibb_sampling": True,
            "simulation_ratio": "1",
            "substrate_ratio": "2/10",
            "output_LDOS": True,
            "output_ff": False,
            "geometry": "balls",
            "material_function": None,
            "polarization_in_plane": False
        },
        "structure": {
            "source_position": [0,0,0],
            "distance": 0.03,
            "radius": 0.035,
            "pyramid_height": 0.2,
            "pyramid_width": 0.12,
            "truncation_width": 0.1,
            "CL_thickness": 0.1,
            "CL_material":"Ag_visible",
            "source_direction": [0,0,1],
            "source_on_wall": False,
            "frequency_center": 2.5,
            "frequency_width": 1.33,
            "number_of_freqs": 8,
            "cutoff": 4
        },
        "result": {}
    }  


if (len(sys.argv)) != 1:
		print("Not enough arguments")
		exit(0)

if simulation == "balls":
    ball = Balls()
    config = config_balls
    ball.setup(config_balls["structure"]) 
    result = ball.simulate(config["simulate"])   
else:
    pyramid = Pyramid()
    config = config_pyramid
    pyramid.setup(config_pyramid["pyramid"])
    result = pyramid.simulate(config["simulate"])
#print(config)
#print(result)
print('Simulation finished at:',datetime.datetime.now())
output_ff = config["simulate"]["output_ff"]
calculate_source_flux = config["simulate"]["calculate_source_flux"]
#data = sim_to_json(config, result,output_ff, calculate_source_flux)
#print('pyramid data:',data)
config["result"] = result
if config["simulate"]["output_ff"] == True:
    config["result"]["fields"] = complex_to_polar_conv(config["result"]["fields"])
write_result("db/test_simulate.json", config)