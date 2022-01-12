import meep as mp	
import numpy as np
import sys
from functionality.functions import *
from functionality.api import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math
import time

from meep.materials import Al, GaN, Au, Ag, Si3N4

###Execute simulation changes pythonpath, this is a work around
try:
	from structure.simstruct import SimStruct
except ImportError:
	from simstruct import SimStruct

#Defines the Pyramid class that inherits from SimStruct. Pyramid class defines all calculations, functions used by pyramid simulations only.

class Pyramid(SimStruct):
#Parameters for the pyramid
	def setup(self, config, sim_config, debug = False):

		#Read pyramid data from config file
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

		#Read simulation data from config file
		resolution = sim_config["resolution"]
		simulation_ratio = eval(sim_config["simulation_ratio"])
		substrate_ratio = eval(sim_config["substrate_ratio"])
		padding = sim_config["padding"]
		dpml = sim_config["dpml"]
		geometry = sim_config["geometry"]
		material_functions=sim_config["material_function"]
		polarization_in_plane = sim_config["polarization_in_plane"]

		substrate_height=self.pyramid_height*substrate_ratio	#height of the substrate, measured as fraction of pyramid height
		#"Cell size"
		sx=self.pyramid_width*simulation_ratio					#size of the cell in xy-plane is measured as a fraction of pyramid width
		sy=sx
		sz=self.pyramid_height*simulation_ratio					#z-"height" of sim. cell measured as a fraction of pyramid height.	

		sh=substrate_height
		padding=padding											##distance from pml_layers to flux regions so PML don't overlap flux regions


		self.sx, self.sy, self.sz, self.sh = sx,sy,sz,sh

		#"Material parameters" 
		air = mp.Medium(epsilon=1)					#air dielectric value
		SiC = mp.Medium(epsilon=2.6465**2)
		SubstrateEps = SiC				#substrate epsilon

		#Maps string to mp.Material object
		if self.CL_material == "Au":
			self.CL_material = Au
		elif self.CL_material == "Ag":
			self.CL_material = Ag
		else:
			self.CL_material = GaN
		print('CL MATERIAL:', self.CL_material, self.CL_material)


		#"Geometry to define the substrate and block of air to truncate the pyramid if self.truncation =/= 0"
		self.geometries=[]
		"Creates a substrate"
		self.geometries.append(mp.Block(center=mp.Vector3(0,0,sz/2-sh/2+dpml/2),
					size=mp.Vector3(2*sx+2*dpml,2*sy+2*dpml,sh+dpml),
					material=SubstrateEps))
		if geometry == False:
			self.geometries = []
		print('Using geometries: ', self.geometries, 'argument: ', geometry)



		#Logic to truncate the pyramid
		if (self.truncation_width > 0):
			#calculate height of truncation. i.e height of the pyramid that gets removed when truncated.
			truncation_height = (self.truncation_width/2)*math.tan((math.pi*62)/180)
			print('trunc h,w', truncation_height, self.truncation_width)
		else:
			truncation_height = 0

		#Logic to insert capping layer over the pyramid
		if (self.CL_thickness > 0):
			coating = True
			#TODO: Make pyramid angle an input in sim_spec.TODO: for th_tot, is it correct to take - CL_thickness (?)
			#increase size of pyramid to account for metal coating. Easier calculating material function this way.
			pyramid_width_tot = self.pyramid_width + 2*self.CL_thickness 
			pyramid_height_tot = self.pyramid_height + self.CL_thickness*math.tan((math.pi*62)/180)
			truncation_height_tot = truncation_height + self.CL_thickness*math.tan((math.pi*62)/180)-self.CL_thickness
		else:
			coating = False

		# "Function for creating pyramid"

		"Different functions for creating the pyramid. The functions take a vector and returns the material at that points. Loops through the entire simulation cell"


		#paints out a hexagon with the help of 4 straight lines in the big if statement
		#here, v is measured from center to vertice. h is measured from center to edge.
		def isInsidexy2(vec):
			while (vec.z <= sz/2-sh and vec.z >= sz/2-sh-self.pyramid_height):
				v=(self.pyramid_width/(2*self.pyramid_height))*vec.z+(self.pyramid_width/(2*self.pyramid_height))*(sh+self.pyramid_height-sz/2)
				h=math.cos(math.pi/6)*v
				k=1/(2*math.cos(math.pi/6))
				if (-h<=vec.x<=h and vec.y <= k*vec.x+v and vec.y <= -k*vec.x+v and vec.y >= k*vec.x-v and vec.y >= -k*vec.x-v):
					return GaN
				else:
					return air
			else:
				return air


		#function to create truncated pyramid with metal coating on top
		def truncPyramidWithCoating(vec):
			#TODO: Optimize function
			while (vec.z <= sz/2-sh and vec.z >= sz/2-sh-pyramid_height_tot+truncation_height_tot):
				v=(pyramid_width_tot/(2*pyramid_height_tot))*vec.z+(pyramid_width_tot/(2*pyramid_height_tot))*(sh+pyramid_height_tot-sz/2)
				h=math.cos(math.pi/6)*v
				k=1/(2*math.cos(math.pi/6))
				v_inner = v - self.CL_thickness
				h_inner = h - self.CL_thickness
				if (-h<=vec.x<=h and vec.y <= k*vec.x+v and vec.y <= -k*vec.x+v and vec.y >= k*vec.x-v and vec.y >= -k*vec.x-v):
					while (vec.z >= sz/2-sh-self.pyramid_height+truncation_height):
						if (-h_inner<=vec.x<=h_inner and vec.y <= k*vec.x+v_inner and vec.y <= -k*vec.x+v_inner and vec.y >= k*vec.x-v_inner and vec.y >= -k*vec.x-v_inner):
							#inner pyramid, inside capping layer
							return GaN
						else:
							return self.CL_material
					return self.CL_material
				else:
					return air
			else:
				return air

		materialFunction = None
		if material_functions == "truncPyramidWithCoating":
			self.materialFunction = truncPyramidWithCoating
		elif material_functions == "isInsidexy2":
			self.materialFunction = isInsidexy2
		else:
			self.materialFunction = None
		print('materialfunction: ',self.materialFunction, 'using argument: ',material_functions)

		inner_pyramid_height = pyramid_height_tot - self.CL_thickness

		#Assuming a (MC_thickness) nm capping layer, (that is included in the total pyramid height and width)
		#I can assume the inner pyramid to be pyramid height - 100 nm and pyramid width - 2*100 nm

		#Logic to place source on wall. Not sure if this logic works any longer. Needs to be verified.
		if self.source_on_wall == True:
			if self.truncation_width > 0:
				inner_pyramid_height = self.pyramid_height*(1-self.truncation_width/self.pyramid_width)
			#if source is placed on wall we need to convert position to MEEP coordinates
			abs_source_position_x = (inner_pyramid_height*self.source_position[2]*math.cos(math.pi/6))/math.tan(62*math.pi/180)
			abs_source_position_y = self.source_position[1]*math.tan(math.pi/6)*(inner_pyramid_height*self.source_position[2]*math.cos(math.pi/6))/math.tan(62*math.pi/180)
			#abs_source_position_z=sz/2-sh-inner_pyramid_height+inner_pyramid_height*(self.source_position[2])
			abs_source_position_z = sz/2-sh-inner_pyramid_height + self.source_position[2]
			print('spos with source_on_wall:',abs_source_position_x,abs_source_position_y,abs_source_position_z)	
		else:
			#else it is assumed source is placed on top of pyramid
			#TODO: implement function that can take of case with both truncated and non-truncated pyramid AND move source in x,y on truncated source
			if self.truncation_width > 0:
				inner_pyramid_height = self.pyramid_height-truncation_height
			abs_source_position_x = 0
			abs_source_position_y = 0
			print('inn pyr, ph, th, sz/2, sh, szpos', inner_pyramid_height, self.pyramid_height, truncation_height, sz/2, sh, self.source_position[2])
			abs_source_position_z = sz/2-sh-inner_pyramid_height+self.source_position[2]
			print('spos with source_on_top:',abs_source_position_x,abs_source_position_y,abs_source_position_z)
			if True:
				abs_source_position_z = abs_source_position_z + 1/(2*resolution) #add 0.5 pixel to truncate source 1 pixel down

		self.abs_source_position_x, self.abs_source_position_y, self.abs_source_position_z = abs_source_position_x, abs_source_position_y, abs_source_position_z

		if polarization_in_plane == True: 
			#change source direction to be parallell with pyramid wall.
			self.source_direction = PolarizeInPlane(self.source_direction,self.pyramid_height,self.pyramid_width)
			print('new source dir',self.source_direction)

		

if __name__ == "__main__":
	if (len(sys.argv)) != 2:
		print("Not enough arguments")
		exit(0)
	config = read("db/tmp/tmp.json")
	if config == None:
		print("could not open tmp/tmp.json for simulation")
		exit(0)
	pyramid = Pyramid()
	pyramid.setup(config["pyramid"], config["simulate"])
	result = pyramid.simulate(config["simulate"])
	output_ff = config["simulate"]["output_ff"]
	config["result"] = result
	if output_ff == True:
		config["result"]["fields"] = complex_to_polar_conv(config["result"]["fields"])
	write_result("db/initial_results/{}.json".format(sys.argv[1]), config)
