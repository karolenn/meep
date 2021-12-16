import meep as mp	
import numpy as np
import sys
from functionality.functions import *
from functionality.api import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math
import time

###Execute simulation changes pythonpath, this is a work around
try:
	from structure.simstruct import SimStruct
except ImportError:
	from simstruct import SimStruct

###PARAMETERS FOR THE SIMULATION and PYRAMID##############################################


class Pyramid(SimStruct):
#Parameters for the pyramid
	def setup(self, config, debug = False):
		self.source_position = config["source_position"] 
		self.source_on_wall = config["source_on_wall"]
		self.pyramid_height = config["pyramid_height"] 
		self.pyramid_width = config["pyramid_width"] 
		self.frequency_center = config["frequency_center"] 
		self.frequency_width = config["frequency_width"] 
		self.number_of_freqs = config["number_of_freqs"] 
		self.cutoff = config["cutoff"]
		self.debug = debug
		self.source_direction = config["source_direction"]
		self.truncation_width = config["truncation_width"]
		self.CL_thickness = config["CL_thickness"]
		self.CL_material = config["CL_material"]



if __name__ == "__main__":
	if (len(sys.argv)) != 2:
		print("Not enough arguments")
		exit(0)
	config = read("db/tmp/tmp.json")
	if config == None:
		print("could not open tmp/tmp.json for simulation")
		exit(0)
	pyramid = Pyramid()
	pyramid.setup(config["pyramid"])
	result = pyramid.simulate(config["simulate"])
	output_ff = config["simulate"]["output_ff"]
	data = sim_to_json(config, result,output_ff)
	#print('pyramid data:',data)
	write_result("db/initial_results/{}.json".format(sys.argv[1]), data)
	#for k,v in data["result"].items():
	#	print(k, v)
	
	# if (len(sys.argv)) != 7:
	# 	print("Not enough arguments")
	# 	exit(0)

	# "1 unit distance is meep is thought of as 1 micrometer here."

	# "Structure Geometry"
	# resolution= int(sys.argv[1])					#resolution of the pyramid. Measured as number of pixels / unit distance
	# simulation_time=int(sys.argv[2])				#simulation time for the sim. #Multiply by a and divide by c to get time in fs.
	# source_position=float(sys.argv[3])					#pos of source measured	measured as fraction of tot. pyramid height from top. 
	# pyramid_height=float(sys.argv[4])				#height of the pyramid in meep units 3.2
	# pyramid_width=float(sys.argv[5])					#width measured from edge to edge 2.6
	# dpml=float(sys.argv[6])
	# pyramid = Pyramid()
	# pyramid.simulate(resolution, simulation_time, source_position, pyramid_height, pyramid_width, dpml)
	
