from structure.pyramid import Pyramid
import sys
import datetime
from functionality.api import polar_to_complex_conv,sim_to_json,write_result

#FOR RUNNING ONE SIMULATION WITH THE CONFIG BELOW

config = {
       "simulate": {
            "resolution": 60,
            "use_fixed_time": False,
            "simulation_time": 5,
            "dpml": 0.1,
            "padding": 0.025,
            "ff_pts": 800,
            "ff_calc": "Below",
            "ff_cover": False,
            "use_symmetries": True,
            "calculate_flux": True,
            "ff_calculations": True,
            "ff_angle": 6,
            "fibb_sampling": True,
            "simulation_ratio": "6/5",
            "substrate_ratio": "1/10",
            "output_ff": True,
            "polarization_in_plane": False
        },
        "pyramid": {
            "source_position": (0,0,0.02),
            "pyramid_height": 0.814,
            "pyramid_width": 1,
            "truncation_width": 0.1,
            "CL_thickness": 0.1,
            "source_direction": (1,0,0),
            "source_on_wall": False,
            "frequency_center": 2,
            "frequency_width": 0.3,
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
print(result[0],result[1],result[2])
#print(result[3])
print('Simulation finished at:',datetime.datetime.now())
output_ff = config["simulate"]["output_ff"]
data = sim_to_json(config, result,output_ff)
#print('pyramid data:',data)
write_result("db/test_simulate.json", data)