import meep as mp	
import numpy as np
from utils.functions import *
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import math as math
import time

###PARAMETERS FOR THE SIMULATION##############################################

#from mpl_toolkits.mplot3d import Axes3D


class Pyramid():

	def setup(self, config, debug = False):
		self.source_position = config["source_position"] 
		self.pyramid_height = config["pyramid_height"] 
		self.pyramid_width = config["pyramid_width"] 
		self.frequency_center = config["frequency_center"] 
		self.frequency_width = config["frequency_width"] 
		self.number_of_freqs = config["number_of_freqs"] 
		self.cutoff = config["cutoff"]
		self.debug = debug
		self.source_direction = eval(config["source_direction"])

	def print(self,*args):
		if self.debug:
			print(args)
	 	
	def create_symetry(self):
		if self.source_direction ==	mp.Ex:
			symmetry=[mp.Mirror(mp.Y),mp.Mirror(mp.X,phase=-1)]
			self.print('symmetry:','Ex')
		elif self.source_direction == mp.Ey:
			symmetry=[mp.Mirror(mp.X),mp.Mirror(mp.Y,phase=-1)]
			self.print('symmetry:','Ey')
		elif self.source_direction == mp.Ez:
			symmetry =[mp.Mirror(mp.X)]
			self.print('symmetry:','Ez')
		else:
			symmetry = []
		return symmetry

	#def isInsidexy(vec):
	#		while (vec.z <= sz/2-sh and vec.z >= sz/2-sh-self.pyramid_height):
#
#				#h=pyramid_width/2+vec.z*(pyramid_width/(2*(sz/2-sh)))
	#			h=self.pyramid_width/(2*self.pyramid_height)*vec.z-(self.pyramid_width/(2*self.pyramid_height))*(sz/2-sh-self.pyramid_height)
	#			v=h*math.tan(math.pi/6)	
	#			center=mp.Vector3(h,v,0)
	#			q2x = math.fabs(vec.x - center.x+h) #transform the test point locally and to quadrant 2m nm
	#			q2y = math.fabs(vec.y - center.y+v) # transform the test point locally and to quadrant 2
	#			if (q2x > h) or (q2y > v*2):
	#				return air
	#			if  ((2*v*h - v*q2x - h*q2y) >= 0): #finally the dot product can be reduced to this due to the hexagon symmetry
	#				return GaN
	#			else:
	#				return air 
	#		else:
	#			return air


	def simulate(self, config):
		dpml = config["dpml"]
		resolution = config["resolution"]
		use_fixed_time = config["use_fixed_time"]
		simulation_time = config["simulation_time"]
		padding = config["padding"]
		ff_pts = config["ff_pts"]
		ff_cover = config["ff_cover"]
		use_symmetries = config["use_symmetries"]
		calculate_flux = config["calculate_flux"]
		ff_calculations = config["ff_calculations"]
		ff_angle = math.pi/config["ff_angle"]
		simulation_ratio = eval(config["simulation_ratio"])
		substrate_ratio = eval(config["substrate_ratio"])

		substrate_height=self.pyramid_height/20				#height of the substrate, measured as fraction of pyramid height
		#"Cell size"
		sx=self.pyramid_width*(6/5)						#size of the cell in xy-plane is measured as a fraction of pyramid width
		sy=sx
		sz=self.pyramid_height*(6/5)						#z-"height" of sim. cell measured as a fraction of pyramid height.		
		sh=substrate_height

		padding=padding							##distance from pml_layers to flux regions so PML don't overlap flux regions
		#"Inarguments for the simulation"
		cell=mp.Vector3(sx+2*dpml,sy+2*dpml,sz+2*dpml)	 		#size of the simulation cell in meep units
		#"Direction for source"
										#symmetry has normal in x & y-direction, phase-even z-dir 
		#"Frequency parameters for gaussian source"
		#"Calculation and plotting parameters"
								#calculate flux at angle measured from top of the pyramid

		def define_flux_regions(sx, sy, sz, padding):
			fluxregion = []
			fluxregion.append(mp.FluxRegion(					#region x to calculate flux from
				center=mp.Vector3(sx/2-padding,0,0),
				size=mp.Vector3(0,sy-padding*2,sz-padding*2),
				direction=mp.X))

			fluxregion.append(mp.FluxRegion(					# region -x to calculate flux from
				center=mp.Vector3(-sx/2+padding,0,0),
				size=mp.Vector3(0,sy-padding*2,sz-padding*2),
				direction=mp.X,
				weight=-1))

			fluxregion.append(mp.FluxRegion(					#region y to calculate flux from
				center=mp.Vector3(0,sy/2-padding,0),
				size=mp.Vector3(sx-padding*2,0,sz-padding*2),
				direction=mp.Y))

			fluxregion.append(mp.FluxRegion(					#region -y to calculate flux from
				center=mp.Vector3(0,-sy/2+padding,0),
				size=mp.Vector3(sx-padding*2,0,sz-padding*2),
				direction=mp.Y,
				weight=-1))

			fluxregion.append(mp.FluxRegion(					#z-bottom region to calculate flux from
				center=mp.Vector3(0,0,sz/2-padding),
				size=mp.Vector3(sx-padding*2,sy-padding*2,0),
				direction=mp.Z))

			fluxregion.append(mp.FluxRegion(					#z-top region to calculate flux from
				center=mp.Vector3(0,0,-sz/2+padding),
				size=mp.Vector3(sx-padding*2,sy-padding*2,0),
				direction=mp.Z,
				weight=-1))
			return fluxregion

		def define_nearfield_regions(sx, sy, sz, sh, padding, ff_cover):
			nearfieldregions=[]
			if ff_cover == False:
				nearfieldregions.append(mp.Near2FarRegion(
					center=mp.Vector3(sx/2-padding,0,-sh/2),
					size=mp.Vector3(0,sy-padding*2,sz-sh-padding*2),
					direction=mp.X))

				nearfieldregions.append(mp.Near2FarRegion(
					center=mp.Vector3(-sx/2+padding,0,-sh/2),
					size=mp.Vector3(0,sy-padding*2,sz-sh-padding*2),
					direction=mp.X,
					weight=-1))

				nearfieldregions.append(mp.Near2FarRegion(
					center=mp.Vector3(0,sy/2-padding,-sh/2),
					size=mp.Vector3(sx-padding*2,0,sz-sh-padding*2),
					direction=mp.Y))

				nearfieldregions.append(mp.Near2FarRegion(
					center=mp.Vector3(0,-sy/2+padding,-sh/2),
					size=mp.Vector3(sx-padding*2,0,sz-sh-padding*2),
					direction=mp.Y,
					weight=-1))
		#under the substrate
		#nearfieldregions.append(mp.Near2FarRegion(
			#	center=mp.Vector3(0,0,sz/2-padding),
			#	size=mp.Vector3(sx-padding*2,sy-padding*2,0),
			#	direction=mp.Z))

				nearfieldregions.append(mp.Near2FarRegion(					#nearfield -z. above pyramid.		
					center=mp.Vector3(0,0,-sz/2+padding),
					size=mp.Vector3(sx-padding*2,sy-padding*2,0),
					direction=mp.Z,
					weight=-1))
			else:
				nearfieldregions.append(mp.Near2FarRegion(
					center=mp.Vector3(sx/2-padding,0,0),
					size=mp.Vector3(0,sy-padding*2,sz-padding*2),
					direction=mp.X))

				nearfieldregions.append(mp.Near2FarRegion(
					center=mp.Vector3(-sx/2+padding,0,0),
					size=mp.Vector3(0,sy-padding*2,sz-padding*2),
					direction=mp.X,
					weight=-1))

				nearfieldregions.append(mp.Near2FarRegion(
					center=mp.Vector3(0,sy/2-padding,0),
					size=mp.Vector3(sx-padding*2,0,sz-padding*2),
					direction=mp.Y))

				nearfieldregions.append(mp.Near2FarRegion(
					center=mp.Vector3(0,-sy/2+padding,0),
					size=mp.Vector3(sx-padding*2,0,sz-padding*2),
					direction=mp.Y,
					weight=-1))
		#under the substrate
				nearfieldregions.append(mp.Near2FarRegion(
					center=mp.Vector3(0,0,sz/2-padding),
					size=mp.Vector3(sx-padding*2,sy-padding*2,0),
					direction=mp.Z))

				nearfieldregions.append(mp.Near2FarRegion(					#nearfield -z. above pyramid.		
					center=mp.Vector3(0,0,-sz/2+padding),
					size=mp.Vector3(sx-padding*2,sy-padding*2,0),
					direction=mp.Z,
					weight=-1))
			return nearfieldregions

					

		###GEOMETRY FOR THE SIMULATION#################################################

		#"Material parameters"
		GaN = mp.Medium(epsilon=5.76)					#GaN n^2=epsilon, n=~2.4 
		air = mp.Medium(epsilon=1)					#air dielectric value
		SubstrateEps = mp.Medium(epsilon=5.76)				#substrate epsilon

		#"Geometry to define the Substrate"

		Substrate=[mp.Block(center=mp.Vector3(0,0,sz/2-sh/2+dpml/2),
					size=mp.Vector3(sx+2*dpml,sy+2*dpml,sh+dpml),
					material=SubstrateEps)]

		###SYMMETRIES#########################################################
		#"Symmetry logic."
		if use_symmetries:
			symmetry = self.create_symetry()

		# "Function for creating pyramid"

		# "isInsidexy checks if a point is inside the pyramid or not. If true, sets epsilon to 5.76, if false, sets it to 1. "
		# "the function loops over the height of the pyramid in z-direction. It transform the test point to vertice 2. It then performs 3 test to check if the point"
		# "is inside a hexagonal area or not. Decreasing/increasing h over the span of z then forms a pyramid. Courtesy to playchilla.com for logic test if a point is inside a hexagon or not."
		# "loops over the height of the pyramid in z-direction. "
		def isInsidexy(vec):
			while (vec.z <= sz/2-sh and vec.z >= sz/2-sh-self.pyramid_height):

				#h=pyramid_width/2+vec.z*(pyramid_width/(2*(sz/2-sh)))
				h=self.pyramid_width/(2*self.pyramid_height)*vec.z-(self.pyramid_width/(2*self.pyramid_height))*(sz/2-sh-self.pyramid_height)
				v=h*math.tan(math.pi/6)	
				center=mp.Vector3(h,v,0)
				q2x = math.fabs(vec.x - center.x+h) #transform the test point locally and to quadrant 2m nm
				q2y = math.fabs(vec.y - center.y+v) # transform the test point locally and to quadrant 2
				if (q2x > h) or (q2y > v*2):
					return air
				if  ((2*v*h - v*q2x - h*q2y) >= 0): #finally the dot product can be reduced to this due to the hexagon symmetry
					return GaN
				else:
					return air 
			else:
				return air


		###PML_LAYERS###################################################################

		pml_layer=[mp.PML(dpml)]

		###SOURCE#######################################################################

		#"A gaussian with pulse source proportional to exp(-iwt-(t-t_0)^2/(2w^2))"

		#"Source position"
		abs_source_position=sz/2-sh-self.pyramid_height+self.pyramid_height*(self.source_position)
		source=[mp.Source(mp.GaussianSource(frequency=self.frequency_center,fwidth=self.frequency_width, cutoff=self.cutoff),	#gaussian current-source
				component=self.source_direction,
				center=mp.Vector3(0,0,abs_source_position))]

		sim=mp.Simulation(cell_size=cell,
				geometry=Substrate,
				symmetries=symmetry,
				sources=source,
				dimensions=3,
				material_function=isInsidexy,
				boundary_layers=pml_layer,
				split_chunks_evenly=False,
				resolution=resolution)

		###REGIONS######################################################################

		#"These regions define the borders of the cell with distance 'padding' between the flux region and the dpml region to avoid calculation errors."
		if calculate_flux:
			flux_regions = define_flux_regions(sx,sy,sz,padding)
			fr1,fr2, fr3, fr4, fr5, fr6 = flux_regions

		###FIELD CALCULATIONS###########################################################
			#"Tells meep with the function 'add_flux' to collect and calculate the flux in the corresponding regions and put them in a flux data object"
		flux_total=sim.add_flux(self.frequency_center, self.frequency_width,self.number_of_freqs,fr1,fr2, fr3, fr4, fr5, fr6 )	#calculate flux for flux regions
			#flux_data_tot=sim.get_flux_data(flux_total)					#get flux data for later reloading

		###FAR FIELD REGION#############################################################

		#"The simulation calculates the far field flux from the regions 1-5 below. It correspons to the air above and at the side of the pyramids. The edge of the simulation cell that touches the substrate is not added to this region. Far-field calculations can not handle different materials."
		nearfieldregions = define_nearfield_regions(sx, sy, sz, sh, padding, ff_cover)
		if ff_cover == False:
			nfr1, nfr2, nfr3, nfr4, nfr6 = nearfieldregions
			nearfield=sim.add_near2far(self.frequency_center,self.frequency_width,self.number_of_freqs,nfr1 ,nfr2, nfr3, nfr4, nfr6)

		else:
			nfr1, nfr2, nfr3, nfr4, nfr5, nfr6 = nearfieldregions				
			nearfield=sim.add_near2far(self.frequency_center,self.frequency_width,self.number_of_freqs,nfr1 ,nfr2, nfr3, nfr4, nfr5, nfr6)
		###RUN##########################################################################
		#"The run constructor for meep."
		if use_fixed_time:
			sim.run(
			#mp.at_beginning(mp.output_epsilon),
			#until_after_sources=mp.stop_when_fields_decayed(2,mp.Ey,mp.Vector3(0,0,sbs_cource_position+0.2),1e-2))
			until=simulation_time)
		else:
			sim.run(
			#mp.at_beginning(mp.output_epsilon),
			until_after_sources=mp.stop_when_fields_decayed(2,self.source_direction,mp.Vector3(0,0,abs_source_position+0.2),1e-3))

		###OUTPUT CALCULATIONS##########################################################

		#"Calculate the poynting flux given the far field values of E, H."
		myIntegration = True
		nfreq=self.number_of_freqs
		r=2*math.pow(self.pyramid_height,2)*self.frequency_center*2*10 				# 10 times the Fraunhofer-distance
		if ff_calculations:
			P_tot_ff = np.zeros(self.number_of_freqs)
									
			npts=ff_pts							#number of far-field points
			Px=0
			Py=0
			Pz=0
			theta=ff_angle
			phi=math.pi*2
			#"How many points on the ff-sphere"
			range_npts=int((theta/math.pi)*npts)
			global xPts
			xPts=[]

			global yPts
			yPts=[]

			global zPts
			zPts=[]

			if myIntegration == True:
				#how to pick ff-points, this uses fibbonaci-sphere distribution
				fibspherepts(r,theta,npts,xPts,yPts,zPts)
				npts=range_npts

				for n in range(npts):

					ff=sim.get_farfield(nearfield, mp.Vector3(xPts[n],yPts[n],zPts[n]))
					i=0
					for k in range(nfreq):
						"Calculate the poynting vector in x,y,z direction"
						Pr = myPoyntingFlux(ff,i)	
						"the spherical cap has area 2*pi*r^2*(1-cos(theta))"
						"divided by npts and we get evenly sized area chunks" 					
						surface_Element=2*math.pi*pow(r,2)*(1-math.cos(theta))/range_npts
						P_tot_ff[k] += surface_Element*(1)*(Pr)
						i = i + 6 #to keep track of the correct entries in the ff array
						#use meeps integration




					#"P_tot_ff[k] is calculated for each freq. Now the loop should make the spacing between two points larger the further down the sphere we goes and surface_elements might overlap here. Check in the future the errors."


			##CALCULATE FLUX OUT FROM BOX###########################################
		if calculate_flux:
			#"Initialize variables to be used, pf stands for 'per frequency'"
			flux_tot_value = np.zeros(self.number_of_freqs)						#total flux out from box
			flux_tot_ff_ratio = np.zeros(self.number_of_freqs)						
			flux_tot_out = mp.get_fluxes(flux_total)			#save total flux data

			if ff_calculations:
				#"the for loop sums up the flux for all frequencies and stores it in flux_tot_value and flux_top_value"
				#"it also calculates the ratio betwewen top and total flux out for all frequencies and stores it in an array 'flux_tot_pf_ratio'"
				ff_freqs = mp.get_near2far_freqs(nearfield)

				for i in range(self.number_of_freqs):

					flux_tot_ff_ratio[i] =P_tot_ff[i]/flux_tot_out[i]			#sums up the total flux out
				self.print('Total_Flux:',flux_tot_out,'Flux_ff:',P_tot_ff,'ratio:',flux_tot_ff_ratio,'sim_time:',simulation_time,'dpml:',dpml,'res:',resolution,'source_position:',self.source_position,'p_height:',self.pyramid_height,'p_width:',self.pyramid_width,'freqs:', ff_freqs)
				return flux_tot_out, list(P_tot_ff), list(flux_tot_ff_ratio), ff_freqs

			else:
				self.print('Total Flux:',flux_tot_out,'ff_flux:',None,'simulation_time:',simulation_time,'dpml:',dpml,'res:',resolution,'r:',r,'res_ff:',None , 'source_position:',self.source_position)
				return flux_tot_out, None , None, 





	###OUTPUT DATA##################################################################


if __name__ == "__main__":
	if (len(sys.argv)) != 7:
		print("Not enough arguments")
		exit(0)

	"1 unit distance is meep is thought of as 1 micrometer here."

	"Structure Geometry"
	resolution= int(sys.argv[1])					#resolution of the pyramid. Measured as number of pixels / unit distance
	simulation_time=int(sys.argv[2])				#simulation time for the sim. #Multiply by a and divide by c to get time in fs.
	source_position=float(sys.argv[3])					#pos of source measured	measured as fraction of tot. pyramid height from top. 
	pyramid_height=float(sys.argv[4])				#height of the pyramid in meep units 3.2
	pyramid_width=float(sys.argv[5])					#width measured from edge to edge 2.6
	dpml=float(sys.argv[6])
	pyramid = Pyramid()
	pyramid.simulate(resolution, simulation_time, source_position, pyramid_height, pyramid_width, dpml)
	
