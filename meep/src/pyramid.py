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
	def __init__(self, \
				 source_pos, \
				 pyramid_height, \
				 pyramid_width, \
				 use_symmetries = True, \
				 calculate_flux = True, \
				 far_field_calculation = True, \
				 debug=False, \
				 source_direction=mp.Ey, \
				 ):
		self.source_pos = source_pos
		self.pyramid_height = pyramid_height
		self.pyramid_width = pyramid_width
		self.use_symmetries = use_symmetries
		self.calculate_flux = calculate_flux
		self.far_field_calculation = far_field_calculation
		self.source_direction = source_direction

		self.debug = debug

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

	def isInsidexy(vec, pyramid_width, pyramid_height, sz, sh):
			while (vec.z <= sz/2-sh and vec.z >= sz/2-sh-pyramid_height):

				#h=pyramid_width/2+vec.z*(pyramid_width/(2*(sz/2-sh)))
				h=pyramid_width/(2*pyramid_height)*vec.z-(pyramid_width/(2*pyramid_height))*(sz/2-sh-pyramid_height)
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

	def define_flux_regions(self, sx, sy, sz, padding):
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
		return tuple(fluxregion)


	def simulate(self, resolution, simulation_time, dpml):

		substrate_height=self.pyramid_height/20				#height of the substrate, measured as fraction of pyramid height
		#"Cell size"
		sx=self.pyramid_width*(6/5)						#size of the cell in xy-plane is measured as a fraction of pyramid width
		sy=sx
		sz=self.pyramid_height*(6/5)						#z-"height" of sim. cell measured as a fraction of pyramid height.		
		padding=0.1							##distance from pml_layers to flux regions so PML don't overlap flux regions
		#"Inarguments for the simulation"
		cell=mp.Vector3(sx+2*dpml,sy+2*dpml,sz+2*dpml)	 		#size of the simulation cell in meep units
		#"Direction for source"
										#symmetry has normal in x & y-direction, phase-even z-dir 
		#"Frequency parameters for gaussian source"
		fcen=2								#center frequency
		df=0.5	   							#frequency span, ranging from 1.7-2.3 
		nfreq=1								#number of frequencies sampled
		#"Calculation and plotting parameters"
		angle = math.pi/6						#calculate flux at angle measured from top of the pyramid

										

		###GEOMETRY FOR THE SIMULATION#################################################

		#"Material parameters"
		GaN = mp.Medium(epsilon=5.76)					#GaN n^2=epsilon, n=~2.4 
		air = mp.Medium(epsilon=1)					#air dielectric value
		SubstrateEps = mp.Medium(epsilon=5.76)				#substrate epsilon

		#sh=substrate_height
		sh=0


		#"Geometry to define the Substrate"

		Substrate=[mp.Block(center=mp.Vector3(0,0,sz/2-sh/2+dpml/2),
					size=mp.Vector3(sx+2*dpml,sy+2*dpml,sh+dpml),
					material=SubstrateEps)]


		###SYMMETRIES#########################################################
		#"Symmetry logic."
		if self.use_symmetries:
			symmetry = self.create_symetry()
		

		# "Function for creating pyramid"

		# "isInsidexy checks if a point is inside the pyramid or not. If true, sets epsilon to 5.76, if false, sets it to 1. "
		# "the function loops over the height of the pyramid in z-direction. It transform the test point to vertice 2. It then performs 3 test to check if the point"
		# "is inside a hexagonal area or not. Decreasing/increasing h over the span of z then forms a pyramid. Courtesy to playchilla.com for logic test if a point is inside a hexagon or not."
		# "loops over the height of the pyramid in z-direction. "
		def isInsidexy(vec):
			return self.isInsidexy(vec, self.pyramid_width, self.pyramid_height, sz, sh)

		###PML_LAYERS###################################################################

		pml_layer=[mp.PML(dpml)]

		###SOURCE#######################################################################

		#"A gaussian with pulse source proportional to exp(-iwt-(t-t_0)^2/(2w^2))"

		#"Source position"
		abs_source_pos=sz/2-sh-self.pyramid_height+self.pyramid_height*(self.source_pos)

		source=[mp.Source(mp.GaussianSource(frequency=fcen,fwidth=df,cutoff=2),	#gaussian current-source
				component=self.source_direction,
				center=mp.Vector3(0,0,0))]

		sim=mp.Simulation(cell_size=cell,
				#geometry=Substrate,
				symmetries=symmetry,
				sources=source,
				dimensions=3,
				#material_function=isInsidexy,
				boundary_layers=pml_layer,
				split_chunks_evenly=False,
				resolution=resolution)

		###REGIONS######################################################################

		#"These regions define the borders of the cell with distance 'padding' between the flux region and the dpml region to avoid calculation errors."
		if self.calculate_flux:
			flux_regions = self.define_flux_regions(sx,sy,sz, padding)
			r1,r2, r3, r4, r5, r6 = flux_regions

		###FIELD CALCULATIONS###########################################################

			#"Tells meep with the function 'add_flux' to collect and calculate the flux in the corresponding regions and put them in a flux data object"
			
			flux_total=sim.add_flux(fcen, df, nfreq, r1, r2, r3, r4, r5, r6 )	#calculate flux for flux regions

			#flux_data_tot=sim.get_flux_data(flux_total)					#get flux data for later reloading

		###FAR FIELD REGION#############################################################

		#"The simulation calculates the far field flux from the regions 1-5 below. It correspons to the air above and at the side of the pyramids. The edge of the simulation cell that touches the substrate is not added to this region. Far-field calculations can not handle different materials."

		nearfieldregion1=mp.Near2FarRegion(
				center=mp.Vector3(sx/2-padding,0,-sh/2),
				size=mp.Vector3(0,sy-padding*2,sz-sh-padding*2),
				direction=mp.X)

		nearfieldregion2=mp.Near2FarRegion(
				center=mp.Vector3(-sx/2+padding,0,-sh/2),
				size=mp.Vector3(0,sy-padding*2,sz-sh-padding*2),
				direction=mp.X,
				weight=-1)

		nearfieldregion3=mp.Near2FarRegion(
				center=mp.Vector3(0,sy/2-padding,-sh/2),
				size=mp.Vector3(sx-padding*2,0,sz-sh-padding*2),
				direction=mp.Y)

		nearfieldregion4=mp.Near2FarRegion(
				center=mp.Vector3(0,-sy/2+padding,-sh/2),
				size=mp.Vector3(sx-padding*2,0,sz-sh-padding*2),
				direction=mp.Y,
				weight=-1)
		#under the substrate
		nearfieldregion5=mp.Near2FarRegion(
				center=mp.Vector3(0,0,sz/2-padding),
				size=mp.Vector3(sx-padding*2,sy-padding*2,0),
				direction=mp.Z)		

		nearfieldregion6=mp.Near2FarRegion(					#nearfield -z. above pyramid.		
				center=mp.Vector3(0,0,-sz/2+padding),
				size=mp.Vector3(sx-padding*2,sy-padding*2,0),
				direction=mp.Z,
				weight=-1)




		nearfield=sim.add_near2far(fcen,df,nfreq,nearfieldregion1,nearfieldregion2,nearfieldregion3,nearfieldregion4,nearfieldregion5,nearfieldregion6)

		###RUN##########################################################################
		#"The run constructor for meep."
		sim.run(
		#mp.at_beginning(mp.output_epsilon),
		#until_after_sources=mp.stop_when_fields_decayed(2,mp.Ey,mp.Vector3(0,0,abs_source_pos+0.2),1e-2))
		until=simulation_time)


		###OUTPUT CALCULATIONS##########################################################

		#"Calculate the poynting flux given the far field values of E, H."

									#how to pick ff-points, this uses fibbonaci-sphere distribution
		r=2*math.pow(self.pyramid_height,2)*fcen*2*10 				# 2 times the Fraunhofer-distance
		if self.far_field_calculation:
			P_tot_ff = np.zeros(nfreq)
									
			npts=1000							#number of far-field points
			Px=0
			Py=0
			Pz=0
			theta=math.pi/4
			phi=math.pi*2
			#"How many points on the ff-sphere"
			range_npts=int((theta/math.pi)*npts)
			global xPts
			xPts=[]

			global yPts
			yPts=[]

			global zPts
			zPts=[]
			#"fibspherepts defined in functions/functions.py"
			fibspherepts(r,theta,npts,xPts,yPts,zPts)
			#print(xPts)

			for n in range(range_npts):


				ff=sim.get_farfield(nearfield, mp.Vector3(xPts[n],yPts[n],zPts[n]))
				#"P_tot_ff calculated as (1/2)Re(E x H*)scalar-product(sin(angleM)cos(angleN),sin(angleM)sin(angleN),-cos(angleM)) where H* is the complex conjugate and this corresponds to the average power density in radial direction from the simulation cell at a distance of r micrometers."

				#"Get the cross product of the poynting flux on the semi-sphere surface. Get the radial magnitude and numerically integrate it "

				i=0
				for k in range(nfreq):
					#"Calculate the poynting vector in x,y,z direction"
					Px=(ff[i+1]*np.conjugate(ff[i+5])-ff[i+2]*np.conjugate(ff[i+4]))
					Px=Px.real
					Py=(ff[i+2]*np.conjugate(ff[i+3])-ff[i]*np.conjugate(ff[i+5]))
					Py=Py.real
					Pz=(ff[i]*np.conjugate(ff[i+4])-ff[i+1]*np.conjugate(ff[i+3]))
					Pz=Pz.real
					#"obtain the radial component of the poynting flux by sqrt(px^2+py^2+pz^2)"
					Pr=math.sqrt(math.pow(Px,2)+math.pow(Py,2)+math.pow(Pz,2))
					#"the spherical cap has area 2*pi*r^2*(1-cos(theta))"
					#"divided by npts and we get evenly sized area chunks" 					
					surface_Element=2*math.pi*pow(r,2)*(1-math.cos(theta))/range_npts
					P_tot_ff[k] += surface_Element*(1)*np.real(Pr)

					i=i+6


					#"P_tot_ff[k] is calculated for each freq. Now the loop should make the spacing between two points larger the further down the sphere we goes and surface_elements might overlap here. Check in the future the errors."


			##CALCULATE FLUX OUT FROM BOX###########################################
		if self.calculate_flux:
			#"Initialize variables to be used, pf stands for 'per frequency'"
			flux_tot_value = np.zeros(nfreq)						#total flux out from box
			flux_tot_ff_ratio = np.zeros(nfreq)						
			flux_tot_out = mp.get_fluxes(flux_total)			#save total flux data

			if self.far_field_calculation:
				#"the for loop sums up the flux for all frequencies and stores it in flux_tot_value and flux_top_value"
				#"it also calculates the ratio betwewen top and total flux out for all frequencies and stores it in an array 'flux_tot_pf_ratio'"
				ff_freqs = mp.get_near2far_freqs(nearfield)

				for i in range(nfreq):

					flux_tot_ff_ratio[i] =P_tot_ff[i]/flux_tot_out[i]			#sums up the total flux out
				self.print('Total_Flux:',flux_tot_out,'Flux_ff:',P_tot_ff,'ratio:',flux_tot_ff_ratio,'sim_time:',simulation_time,'dpml:',dpml,'res:',resolution,'source_pos:',self.source_pos,'p_height:',self.pyramid_height,'p_width:',self.pyramid_width,'freqs:', ff_freqs)
				return flux_tot_out, P_tot_ff, flux_tot_ff_ratio, simulation_time, dpml, resolution, self.source_pos, self.pyramid_height, self.pyramid_width, ff_freqs

			else:
				self. print('Total Flux:',flux_tot_out,'ff_flux:',None,'simulation_time:',simulation_time,'dpml:',dpml,'res:',resolution,'r:',r,'res_ff:',None , 'source_pos:',self.source_pos)
				return flux_tot_out, None , None, simulation_time, dpml, resolution, r, None , self.source_pos, None





	###OUTPUT DATA##################################################################


if __name__ == "__main__":
	if (len(sys.argv)) != 7:
		print("Not enough arguments")
		exit(0)

	"1 unit distance is meep is thought of as 1 micrometer here."

	"Structure Geometry"
	resolution= int(sys.argv[1])					#resolution of the pyramid. Measured as number of pixels / unit distance
	simulation_time=int(sys.argv[2])				#simulation time for the sim. #Multiply by a and divide by c to get time in fs.
	source_pos=float(sys.argv[3])					#pos of source measured	measured as fraction of tot. pyramid height from top. 
	pyramid_height=float(sys.argv[4])				#height of the pyramid in meep units 3.2
	pyramid_width=float(sys.argv[5])					#width measured from edge to edge 2.6
	dpml=float(sys.argv[6])
	pyramid = Pyramid()
	pyramid.simulate(resolution, simulation_time, source_pos, pyramid_height, pyramid_width, dpml)
	