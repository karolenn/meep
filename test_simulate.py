from lib.pyramid import Pyramid
import sys

config = {
        "simulate": {
            "resolution": 60,
            "use_fixed_time": False,
            "simulation_time": 10,
            "dpml": 0.1,
            "padding": 0.025,
            "ff_pts": 1600,
            "ff_calc": "Above",
            "ff_cover": True,
            "use_symmetries": True,
            "calculate_flux": True,
            "ff_calculations": False,
            "ff_angle": 6,
            "simulation_ratio": "6/5",
            "substrate_ratio": "1/10"
        },
        "pyramid": {
            "source_position": 0.01,
            "pyramid_height": 1,
            "pyramid_width": 1,
            "truncation": 0,
            "source_direction": "mp.Ey",
            "frequency_center": 2,
            "frequency_width": 0.5,
            "number_of_freqs": 3,
            "cutoff": 2
        },
        "result": {}
    }  
#print(int(config["source_position"]))

if (len(sys.argv)) != 1:
		print("Not enough arguments")
		exit(0)

#resolution= int(sys.argv[1])					#resolution of the pyramid. Measured as number of pixels / unit distance
#simulation_time=int(sys.argv[2])				#simulation time for the sim. #Multiply by a and divide by c to get time in fs.
#source_pos=float(sys.argv[3])					#pos of source measured	measured as fraction of tot. pyramid height from top. 
#pyramid_height=float(sys.argv[4])				#height of the pyramid in meep units 3.2
#pyramid_width=float(sys.argv[5])					#width measured from edge to edge 2.6
#frequency_center=float(sys.argv[6])
#frequency_width=float(sys.argv[7])
#number_of_freqs=int(sys.argv[8])
#cutoff=int(sys.argv[9])
#dpml=float(sys.argv[10])'

    
pyramid = Pyramid()
pyramid.setup(config["pyramid"])
result = pyramid.simulate(config["simulate"])
print(result)
