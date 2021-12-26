from structure.pyramid import Pyramid
from structure.silverball import Balls
import sys
import datetime
from functionality.api import polar_to_complex_conv,sim_to_json,write_result

#FOR RUNNING ONE SIMULATION WITH THE CONFIG BELOW

config = {
       "simulate": {
            "resolution": 480,
            "use_fixed_time": False,
            "simulation_time": 500,
            "dpml": 0.01,
            "padding": 0.0025,
            "ff_pts": 800,
            "ff_calc": "Below",
            "ff_cover": False,
            "use_symmetries": True,
            "calculate_flux": True,
            "calculate_source_flux": True,
            "source_flux_pixel_size": 2,
            "ff_calculations": False,
            "ff_angle": 6,
            "fibb_sampling": True,
            "simulation_ratio": "1",
            "substrate_ratio": "2/10",
            "output_ff": False,
            "geometry": None,
            "material_function": None,
            "polarization_in_plane": False
        },
        "pyramid": {
            "source_position": [0,0,0],
            "pyramid_height": 0.2,
            "pyramid_width": 0.12,
            "truncation_width": 0.1,
            "CL_thickness": 0.1,
            "CL_material":"Ag",
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

if True:
    ball = Balls()
    ball.setup(config["pyramid"]) 
    result = ball.simulate(config["simulate"])   
else:
    pyramid = Pyramid()
    pyramid.setup(config["pyramid"])
    result = pyramid.simulate(config["simulate"])
print(config)
for i in range(len(result)):
    print(result[i])
#print(result[0],result[1],result[2],result[3])
#print(result[3])
print('Simulation finished at:',datetime.datetime.now())
output_ff = config["simulate"]["output_ff"]
calculate_source_flux = config["simulate"]["calculate_source_flux"]
data = sim_to_json(config, result,output_ff, calculate_source_flux)
#print('pyramid data:',data)
write_result("db/test_simulate.json", data)