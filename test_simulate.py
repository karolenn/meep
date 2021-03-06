from structure.pyramid import Pyramid
import sys
import datetime
from functionality.api import polar_to_complex_conv,sim_to_json,write_result

#FOR RUNNING ONE SIMULATION WITH THE CONFIG BELOW

config = {
       "simulate": {
            "resolution": 60,
            "use_fixed_time": False,
            "simulation_time": 60,
            "dpml": 0.1,
            "padding": 0.025,
            "ff_pts": 60,
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
            "frequency_width": 0.2,
            "number_of_freqs": 2,
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
qw = config["simulate"]["quantum_well"]
data = sim_to_json(config, result,qw)
#print('pyramid data:',data)
write_result("db/test_simulate.json", data)