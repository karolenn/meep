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
	config["result"] = result
	if output_ff == True:
		config["result"]["fields"] = complex_to_polar_conv(config["result"]["fields"])
	write_result("db/initial_results/{}.json".format(sys.argv[1]), config)
