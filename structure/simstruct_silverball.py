import meep as mp	
import numpy as np
import sys
from functionality.functions import *
from functionality.api import *
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import math as math
import time

if True:
	from meep.materials import Al, GaN, Au, Ag, Si3N4, Ag_visible

###PARAMETERS FOR THE SIMULATION and PYRAMID##############################################


class SimStruct_silverball():

#Simulation parameters
	def simulate(self, config):
		start = time.time()
		dpml = config["dpml"]
		resolution = config["resolution"]
		use_fixed_time = config["use_fixed_time"]
		simulation_time = config["simulation_time"]
		padding = config["padding"]
		ff_pts = config["ff_pts"]
		ff_cover = config["ff_cover"]
		ff_calc = config["ff_calc"]
		use_symmetries = config["use_symmetries"]
		calculate_flux = config["calculate_flux"]
		calculate_source_flux = config["calculate_source_flux"]
		pixels = config["source_flux_pixel_size"]
		ff_calculations = config["ff_calculations"]
		ff_angle = math.pi/config["ff_angle"]
		FibonacciSampling = config["fibb_sampling"]
		simulation_ratio = eval(config["simulation_ratio"])
		substrate_ratio = eval(config["substrate_ratio"])
		output_ff = config["output_ff"]
		polarization_in_plane = config["polarization_in_plane"]
		geometry = config["geometry"]
		material_function = config["material_function"]

		substrate_height=self.pyramid_height*substrate_ratio	#height of the substrate, measured as fraction of pyramid height
		#"Cell size"
		sx=self.pyramid_width*simulation_ratio						#size of the cell in xy-plane is measured as a fraction of pyramid width
		sy=sx
		sz=self.pyramid_height*simulation_ratio						#z-"height" of sim. cell measured as a fraction of pyramid height.		
		sh=substrate_height
		sh=0
		padding=padding							##distance from pml_layers to flux regions so PML don't overlap flux regions
		cell=mp.Vector3(sx+2*dpml,sy+2*dpml,sz+2*dpml)	 		#size of the simulation cell in meep units

		if self.CL_material == "Au":
			CL_material = Au
		elif self.CL_material == "Ag":
			CL_material = Ag
		else:
			CL_material = GaN
		print('CL MATERIAL:', CL_material, self.CL_material)

#Symmetries for the simulation	
		def create_symmetry(self):
			if self.source_direction ==	mp.Ex or self.source_direction == [1,0,0]:
				symmetry=[mp.Mirror(mp.Y),mp.Mirror(mp.X,phase=-1)]
			elif self.source_direction == mp.Ey or self.source_direction == [0,1,0]:
				symmetry=[mp.Mirror(mp.X),mp.Mirror(mp.Y,phase=-1)]
			elif self.source_direction == mp.Ez  or self.source_direction == [0,0,1]:
				symmetry =[mp.Mirror(mp.X),mp.Mirror(mp.Y)]
			else:
				symmetry = []
			return symmetry

#Flux regions to calculate the total flux emitted from the simulation
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

#Flux regions to calculate the far field flux
		def define_nearfield_regions(sx, sy, sz, sh, padding, ff_cover):
			nearfieldregions=[]
			nfrAbove=[]
			nfrBelow=[]
			#if ff_calc == True:
			#If we wish to calculate the far-field above and below the pyramid
		#	if ff_calc == "Both":
			nfrAbove.append(mp.Near2FarRegion(
					center=mp.Vector3(sx/2-padding,0,-sh/2),
					size=mp.Vector3(0,sy-padding*2,sz-sh-padding*2),
					direction=mp.X))

			nfrAbove.append(mp.Near2FarRegion(
					center=mp.Vector3(-sx/2+padding,0,-sh/2),
					size=mp.Vector3(0,sy-padding*2,sz-sh-padding*2),
					direction=mp.X,
					weight=-1))

			nfrAbove.append(mp.Near2FarRegion(
					center=mp.Vector3(0,sy/2-padding,-sh/2),
					size=mp.Vector3(sx-padding*2,0,sz-sh-padding*2),
					direction=mp.Y))

			nfrAbove.append(mp.Near2FarRegion(
					center=mp.Vector3(0,-sy/2+padding,-sh/2),
					size=mp.Vector3(sx-padding*2,0,sz-sh-padding*2),
					direction=mp.Y,
					weight=-1))


			nfrAbove.append(mp.Near2FarRegion(					#nearfield -z. above pyramid.		
					center=mp.Vector3(0,0,-sz/2+padding),
					size=mp.Vector3(sx-padding*2,sy-padding*2,0),
					direction=mp.Z,
					weight=-1))

			#Far-field to calculate below transmissions
			nfrBelow.append(mp.Near2FarRegion(
					center=mp.Vector3(sx/2-padding,0,sz/2-sh/2),
					size=mp.Vector3(0,sy-padding*2,sh-padding*2),
					direction=mp.X))

			nfrBelow.append(mp.Near2FarRegion(
					center=mp.Vector3(-sx/2+padding,0,sz/2-sh/2),
					size=mp.Vector3(0,sy-padding*2,sh-padding*2),
					direction=mp.X,
					weight=-1))

			nfrBelow.append(mp.Near2FarRegion(
					center=mp.Vector3(0,sy/2-padding,sz/2-sh/2),
					size=mp.Vector3(sx-padding*2,0,sh-padding*2),
					direction=mp.Y))

			nfrBelow.append(mp.Near2FarRegion(
					center=mp.Vector3(0,-sy/2+padding,sz/2-sh/2),
					size=mp.Vector3(sx-padding*2,0,sh-padding*2),
					direction=mp.Y,
					weight=-1))				
			nfrBelow.append(mp.Near2FarRegion(
					center=mp.Vector3(0,0,sz/2-padding),
					size=mp.Vector3(sx-padding*2,sy-padding*2,0),
					direction=mp.Z))
			if ff_cover == True:
				nfrAbove.append(mp.Near2FarRegion(
					center=mp.Vector3(0,0,sz/2-padding),
					size=mp.Vector3(sx-padding*2,sy-padding*2,0),
					direction=mp.Z))

				nfrBelow.append(mp.Near2FarRegion(
					center=mp.Vector3(0,0,-sz/2-padding),
					size=mp.Vector3(sx-padding*2,sy-padding*2,0),
					direction=mp.Z,
					weight=-1))
			nearfieldregions.append(nfrAbove)
			nearfieldregions.append(nfrBelow)
			return nearfieldregions

		###GEOMETRY FOR THE SIMULATION#################################################

		#"Material parameters" 
		air = mp.Medium(epsilon=1)					#air dielectric value
		SubstrateEps = GaN				#substrate epsilon

		#"Geometry to define the substrate and block of air to truncate the pyramid if self.truncation =/= 0"
		geometries=[]
		#Substrate
		radius_ = 0.035
		distance = 0.03
		geometries.append(mp.Sphere(center=mp.Vector3(0,0,distance/2 + radius_),
					radius = radius_,
					material=CL_material))
		
		geometries.append(mp.Sphere(center=mp.Vector3(0,0,-distance/2  -radius_),
					radius = radius_,
					material=CL_material))

		if geometry == None:
			geometries = None
		
		if material_function == None:
			material_function = None

		###SYMMETRIES#########################################################
		#"Symmetry logic."
		if use_symmetries:
			symmetry = create_symmetry(self)
			print('symmetry on:', symmetry)
		else:
			symmetry = []


		###PML_LAYERS###################################################################

		pml_layer=[mp.PML(dpml)]

		###SOURCE#######################################################################

		#"A gaussian with pulse source proportional to exp(-iwt-(t-t_0)^2/(2w^2))"

		abs_source_position_x = 0
		abs_source_position_y = 0
		abs_source_position_z = 0


		source=[mp.Source(mp.GaussianSource(frequency=self.frequency_center,fwidth=self.frequency_width, cutoff=self.cutoff),	#gaussian current-source
				component=mp.Ez,
				amplitude=1,
				center=mp.Vector3(abs_source_position_x,abs_source_position_y,abs_source_position_z))]
				
		#source.append(mp.Source(mp.GaussianSource(frequency=self.frequency_center,fwidth=self.frequency_width, cutoff=self.cutoff),	#gaussian current-source
		#		component=mp.Ey,
		#		amplitude=self.source_direction[1],
		#		center=mp.Vector3(abs_source_position_x,abs_source_position_y,abs_source_position_z)))

		#source.append(mp.Source(mp.GaussianSource(frequency=self.frequency_center,fwidth=self.frequency_width, cutoff=self.cutoff),	#gaussian current-source
		#		component=mp.Ez,
		#		amplitude=self.source_direction[2],
		#		center=mp.Vector3(abs_source_position_x,abs_source_position_y,abs_source_position_z)))
		#MEEP simulation constructor
		sim=mp.Simulation(cell_size=cell,
				geometry=geometries,
				symmetries=symmetry,
				sources=source,
				#eps_averaging=True,
				#subpixel_tol=1e-4,
				#subpixel_maxeval=1000,
				dimensions=3,
				#default_material=GaN,
				extra_materials=[CL_material],
				boundary_layers=pml_layer,
				split_chunks_evenly=False,
				resolution=resolution)

		###SOURCE REGION###################################################
		print('pixels', pixels)
		def define_flux_source_regions(abs_source_position_x,abs_source_position_y,abs_source_position_z,resolution, pixels):
				distance = pixels*1/resolution
				source_region = []
				source_region.append(mp.FluxRegion(					#region x to calculate flux from
					center=mp.Vector3(abs_source_position_x+distance,abs_source_position_y,abs_source_position_z),
					size=mp.Vector3(0,2*pixels*1/resolution,2*pixels*1/resolution),
					direction=mp.X))

				source_region.append(mp.FluxRegion(					# region -x to calculate flux from
					center=mp.Vector3(abs_source_position_x-distance,abs_source_position_y,abs_source_position_z),
					size=mp.Vector3(0,2*pixels*1/resolution,2*pixels*1/resolution),
					direction=mp.X,
					weight=-1))

				source_region.append(mp.FluxRegion(					#region y to calculate flux from
					center=mp.Vector3(abs_source_position_x,abs_source_position_y+distance,abs_source_position_z),
					size=mp.Vector3(2*pixels*1/resolution,0,2*pixels*1/resolution),
					direction=mp.Y))

				source_region.append(mp.FluxRegion(					#region -y to calculate flux from
					center=mp.Vector3(abs_source_position_x,abs_source_position_y-distance,abs_source_position_z),
					size=mp.Vector3(2*pixels*1/resolution,0,2*pixels*1/resolution),
					direction=mp.Y,
					weight=-1))

				source_region.append(mp.FluxRegion(					#z-bottom region to calculate flux from
					center=mp.Vector3(abs_source_position_x,abs_source_position_y,abs_source_position_z+distance),
					size=mp.Vector3(2*pixels*1/resolution,2*pixels*1/resolution,0),
					direction=mp.Z))

				source_region.append(mp.FluxRegion(					#z-top region to calculate flux from
					center=mp.Vector3(abs_source_position_x,abs_source_position_y,abs_source_position_z-distance),
					size=mp.Vector3(2*pixels*1/resolution,2*pixels*1/resolution,0),
					direction=mp.Z,
					weight=-1))
				return source_region						


		###REGIONS######################################################################

		#"These regions define the borders of the cell with distance 'padding' between the flux region and the dpml region to avoid calculation errors."
		if calculate_flux:
			flux_regions = define_flux_regions(sx,sy,sz,padding)
			fr1,fr2, fr3, fr4, fr5, fr6 = flux_regions
			flux_total=sim.add_flux(self.frequency_center, self.frequency_width,self.number_of_freqs,fr1,fr2, fr3, fr4, fr5, fr6)	#calculate flux for flux regions		

		if calculate_source_flux:
			sr1, sr2, sr3, sr4, sr5, sr6 = define_flux_source_regions(abs_source_position_x,abs_source_position_y,abs_source_position_z,resolution, pixels)
			flux_source=sim.add_flux(self.frequency_center, self.frequency_width,self.number_of_freqs, sr1, sr2, sr3, sr4, sr5, sr6)

		###FAR FIELD REGION#############################################################

		#"The simulation calculates the far field flux from the regions 1-5 below. It correspons to the air above and at the side of the pyramids. The edge of the simulation cell that touches the substrate is not added to this region. Far-field calculations can not handle different materials."
		if ff_calculations == True:
			nearfieldregions = define_nearfield_regions(sx, sy, sz, sh, padding, ff_cover)
			if ff_cover == True:
				nfrA1, nfrA2, nfrA3, nfrA4, nfrA5, nfrA6 = nearfieldregions[0]
				nfrB1, nfrB2, nfrB3, nfrB4, nfrB5, nfrB6 = nearfieldregions[1]
			else:	
				nfrA1, nfrA2, nfrA3, nfrA4, nfrA6 = nearfieldregions[0]
				nfrB1, nfrB2, nfrB3, nfrB4, nfrB6 = nearfieldregions[1]
			if ff_calc == "Both" and ff_cover == True:
				nearfieldAbove=sim.add_near2far(self.frequency_center,self.frequency_width,self.number_of_freqs,nfrA1 ,nfrA2, nfrA3, nfrA4, nfrA5, nfrA6)
				nearfieldBelow=sim.add_near2far(self.frequency_center,self.frequency_width,self.number_of_freqs,nfrB1 ,nfrB2, nfrB3, nfrB4, nfrB5, nfrB6)
			elif ff_calc == "Above" and ff_cover == True:
				nearfieldAbove=sim.add_near2far(self.frequency_center,self.frequency_width,self.number_of_freqs,nfrA1 ,nfrA2, nfrA3, nfrA4, nfrA5, nfrA6)
			elif ff_calc == "Above" and ff_cover == False:
				nearfieldAbove=sim.add_near2far(self.frequency_center,self.frequency_width,self.number_of_freqs,nfrA1 ,nfrA2, nfrA3, nfrA4, nfrA6)
			elif ff_calc == "Below" and ff_cover == True:
				nearfieldBelow=sim.add_near2far(self.frequency_center,self.frequency_width,self.number_of_freqs,nfrB1 ,nfrB2, nfrB3, nfrB4, nfrB5, nfrB6)			
			elif ff_calc == "Below" and ff_cover == False:
				nearfieldBelow=sim.add_near2far(self.frequency_center,self.frequency_width,self.number_of_freqs,nfrB1 ,nfrB2, nfrB3, nfrB4, nfrB6)
			#Assumed ff_calc == Both and ff_cover == False	
			else:
				nearfieldAbove=sim.add_near2far(self.frequency_center,self.frequency_width,self.number_of_freqs,nfrA1 ,nfrA2, nfrA3, nfrA4, nfrA6)
				nearfieldBelow=sim.add_near2far(self.frequency_center,self.frequency_width,self.number_of_freqs,nfrB1 ,nfrB2, nfrB3, nfrB4, nfrB6)
		###RUN##########################################################################
		#"Run the simulation"
		if True:
			sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(0,0*abs_source_position_y,0),size=mp.Vector3(0,sy+2*dpml,sz+2*dpml)))
			#plt.show()
			plt.savefig('foo.pdf')
			sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(0*abs_source_position_x,0,0),size=mp.Vector3(sx+2*dpml,0,sz+2*dpml)))
			
			#plt.show()
			plt.savefig('foo2.pdf')
			sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(0,0,abs_source_position_z),size=mp.Vector3(sx+2*dpml,sy+2*dpml,0)))
			#sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(0,0,sz/2-sh-0.01),size=mp.Vector3(sx+2*dpml,sy+2*dpml,0)))
			plt.savefig('foo3.pdf')


		if use_fixed_time:
			sim.run(
			mp.dft_ldos(self.frequency_center, self.frequency_width, self.number_of_freqs),
		#	mp.at_beginning(mp.output_epsilon),
			#until_after_sources=mp.stop_when_fields_decayed(2,mp.Ey,mp.Vector3(0,0,sbs_cource_position+0.2),1e-2))
			until=simulation_time)
		else:
			#Withdraw maximum dipole amplitude direction
			max_index = self.source_direction.index(max(self.source_direction))
			if max_index == 0:
				detector_pol = mp.Ex
			elif max_index == 1:
				detector_pol = mp.Ey
			else:
				detector_pol = mp.Ez
			#TODO: exchange self.source_direction to maxmimum dipole ampltitude
			sim.run(
			mp.dft_ldos(self.frequency_center, self.frequency_width, self.number_of_freqs),
			#mp.to_appended("ex", mp.at_every(0.6, mp.output_efield_x)),	
			mp.at_beginning(mp.output_epsilon),
			until_after_sources=mp.stop_when_fields_decayed(2,detector_pol,mp.Vector3(0,0,abs_source_position_z+0.2),1e-3))


		###OUTPUT CALCULATIONS##########################################################

		#"Calculate the poynting flux given the far field values of E, H."
		myIntegration = True
		nfreq=self.number_of_freqs
		r=2*math.pow(self.pyramid_height,2)*self.frequency_center*2*10 				# 10 times the Fraunhofer-distance
		if ff_calculations:
			fields = []
			P_tot_ff = np.zeros(self.number_of_freqs)						
			npts=ff_pts							#number of far-field points
			Px=0
			Py=0
			Pz=0
			theta=ff_angle
			phi=math.pi*2
			#"How many points on the ff-sphere"
			#global xPts
			#xPts=[]

			#global yPts
			#yPts=[]

			#global zPts
			#zPts=[]

			Pr_Array=[]

			if myIntegration == True:
				#how to pick ff-points, this uses fibbonaci-sphere distribution
				if FibonacciSampling == True:
					if theta==math.pi/3:
						offset=1.5/npts
					elif theta==math.pi/4:
						npts=npts*2
						offset=1.15/npts
					elif theta==math.pi/5:
						npts=npts*2.5
						offset=0.95/npts
					elif theta==math.pi/6:
						#npts=npts*3
						offset=0.8/npts
					elif theta==math.pi/7:
						npts=npts*3
						offset=0.7/npts
					elif theta==math.pi/8:
						npts=npts*3
						offset=0.6/npts
					elif theta==math.pi/12:
						npts=npts*3
						offset=0.4/npts
					else:
						offset=0.2/npts
					xPts,yPts,zPts = fibspherepts(r,theta,npts,offset)
					range_npts = int((theta/math.pi)*npts)
				else:
					#check out Lebedev quadrature
					#if not fibb sampling then spherical sampling
					theta_pts = npts
					phi_pts = npts*2
					xPts,yPts,zPts = sphericalpts(r,theta,phi,theta_pts,phi_pts)
					range_npts = int(theta_pts*phi_pts + 1)

				if ff_calc == "Both":
					#Pr_ArrayA for above, B for below
					Pr_ArrayA=[]
					Pr_ArrayB=[]
					P_tot_ffA = [0]*(self.number_of_freqs)
					P_tot_ffB = [0]*(self.number_of_freqs)
					for n in range(range_npts):
						ffA=sim.get_farfield(nearfieldAbove, mp.Vector3(xPts[n],yPts[n],zPts[n]))
						ffB=sim.get_farfield(nearfieldBelow, mp.Vector3(xPts[n],yPts[n],-1*zPts[n]))
						i=0
						for k in range(nfreq):
							"Calculate the poynting vector in x,y,z direction"
							PrA = myPoyntingFlux(ffA,i)	
							PrB = myPoyntingFlux(ffB,i)
							Pr_ArrayA.append(PrA)
							Pr_ArrayB.append(PrB)
							"the spherical cap has area 2*pi*r^2*(1-cos(theta))"
							"divided by npts and we get evenly sized area chunks" 					
							surface_Element=2*math.pi*pow(r,2)*(1-math.cos(theta))/range_npts
							P_tot_ffA[k] += surface_Element*(1)*(PrA)
							P_tot_ffB[k] += surface_Element*(1)*(PrB)
							i = i + 6 #to keep track of the correct entries in the ff array
				if ff_calc == "Below":
					for n in range(range_npts):
						ff=sim.get_farfield(nearfieldBelow, mp.Vector3(xPts[n],yPts[n],-1*zPts[n]))
						fields.append({"pos":(xPts[n],yPts[n],zPts[n]),"field":ff})
						i=0
						for k in range(nfreq):
							"Calculate the poynting vector in x,y,z direction"
							Pr = myPoyntingFlux(ff,i)	
							Pr_Array.append(Pr)
							"the spherical cap has area 2*pi*r^2*(1-cos(theta))"
							"divided by npts and we get evenly sized area chunks" 					
							surface_Element=2*math.pi*pow(r,2)*(1-math.cos(theta))/range_npts
							P_tot_ff[k] += surface_Element*(1)*(Pr)
							i = i + 6 #to keep track of the correct entries in the ff array
				if ff_calc == "Above":
					for n in range(range_npts):
						ff=sim.get_farfield(nearfieldAbove, mp.Vector3(xPts[n],yPts[n],zPts[n]))
						fields.append({"pos":(xPts[n],yPts[n],zPts[n]),"field":ff})
				#		print('ff,n,x,y,z',n,xPts[n],yPts[n],zPts[n],ff)
				#		print('fields',fields[n])
				#		print('------')
						i=0
						for k in range(nfreq):
							"Calculate the poynting vector in x,y,z direction"
							Pr = myPoyntingFlux(ff,i)	
							Pr_Array.append(Pr)
							"the spherical cap has area 2*pi*r^2*(1-cos(theta))"
							"divided by npts and we get evenly sized area chunks" 					
							surface_Element=2*math.pi*pow(r,2)*(1-math.cos(theta))/range_npts
							P_tot_ff[k] += surface_Element*(1)*(Pr)
							#print('S',surface_Element,'r',r,'theta',theta,'range npts',range_npts)
							i = i + 6 #to keep track of the correct entries in the ff array
				#	print('P_tot_ff',P_tot_ff)
				#	print('fields',fields)
				#	print('Pr_Array',Pr_Array)

			##CALCULATE FLUX OUT FROM BOX###########################################
		if calculate_flux:
			#"Initialize variables to be used, pf stands for 'per frequency'"
			flux_tot_value = np.zeros(self.number_of_freqs)						#total flux out from box
			flux_tot_ff_ratio = np.zeros(self.number_of_freqs)						
			flux_tot_out = mp.get_fluxes(flux_total)		#save total flux data
			print('freqs',mp.get_flux_freqs(flux_total))
			if calculate_source_flux:
				source_flux_out = mp.get_fluxes(flux_source)


			elapsed_time = round((time.time()-start)/60,1)
			print('LDOS', sim.ldos_data)
			print('LDOS0',sim.ldos_data[0])

			if False:
				fig = plt.figure()
				ax = fig.gca(projection='3d')
				Pr_Max=max(Pr_Array)
				R = [n/Pr_Max  for n in Pr_Array]
				print(R)
	#			R=1
				pts = len(Pr_Array)
				theta,phi = np.linspace(0,np.pi,pts), np.linspace(0,2*np.pi,pts)
				THETA,PHI = np.meshgrid(theta,phi)
				#print(xPts)
				# print(yPts)
				# print(zPts)
				X=np.zeros(pts**2)
				Y=np.zeros(pts**2)
				Z=np.zeros(pts**2)
				print('R pts',pts)
				print('sample pts',len(xPts))
				for n in range(pts):
					xPts[n] = R[n]*xPts[n]
					yPts[n] = R[n]*yPts[n]
					zPts[n] = R[n]*zPts[n]
				# #print(X)
				# #print(Y)
				# #print(Z)
				#ax.set_xlim(-100,100)
				#ax.set_zlim(-100,100)
				#ax.set_ylim(-100,100)

				#ax.scatter(xPts,yPts,zPts)
				u = np.linspace(0, 2 * np.pi, 120)
				v = np.linspace(0, np.pi/6, 120)
				r_ = np.asarray(R)
				x = np.outer(np.cos(u), np.sin(v))
				y = np.outer(np.sin(u), np.sin(v))
				z = R*np.outer(np.ones(np.size(u)), np.cos(v))
				print('lenx',len(x),'leny',len(y),'lenz',len(z))
				#r = np.sqrt(xPts**2+yPts**2+zPts**2)
				#plot(r)
				ax.plot_surface(x,y,z,cmap='plasma')
				#ax.quiver(0,0,0,0,0,1,length=1.2,color='brown',normalize='true')
				#ax.text(0,0,1.3, '$\mathcal{P}$',size=20, zdir=None)
				#ax.set_zlim(-1,1)
				#ax.axis('off')
				#ax.plot_trisurf(list(x),list(y),list(z),cmap='plasma')
				plt.show()
			for n in range(len(flux_tot_out)):
				flux_tot_out[n]=round(flux_tot_out[n],11)
			#	P_tot_ff[n]=round(P_tot_ff[n],9)
			##Some processing to calculate the flux ratios per frequency
			if ff_calculations:
				for n in range(len(flux_tot_out)):
				#	flux_tot_out[n]=round(flux_tot_out[n],9)
					P_tot_ff[n]=round(P_tot_ff[n],11)
				for i in range(self.number_of_freqs):	

					flux_tot_ff_ratio[i] =round(P_tot_ff[i]/flux_tot_out[i],11)		
				if ff_calc == "Both":
					P_tot_ff = []
					P_tot_ff.append(P_tot_ffA)
					P_tot_ff.append(P_tot_ffB)
					flux_tot_ff_ratio = []
					flux_tot_ff_ratioA = [0]*(self.number_of_freqs)	
					flux_tot_ff_ratioB = [0]*(self.number_of_freqs)	
					for i in range(self.number_of_freqs):	
						flux_tot_ff_ratioA[i] =round(P_tot_ffA[i]/flux_tot_out[i],11)		
						flux_tot_ff_ratioB[i] =round(P_tot_ffB[i]/flux_tot_out[i],11)
					flux_tot_ff_ratio.append(flux_tot_ff_ratioA)
					flux_tot_ff_ratio.append(flux_tot_ff_ratioB)				
				#print(fields["pos"])
				if output_ff == False:
					output_ff = 0
				if output_ff:
					return flux_tot_out, list(P_tot_ff), list(flux_tot_ff_ratio), fields, elapsed_time
				elif calculate_source_flux:
					return flux_tot_out, source_flux_out, list(P_tot_ff), list(flux_tot_ff_ratio), elapsed_time
				else:
					return flux_tot_out, list(P_tot_ff), list(flux_tot_ff_ratio), elapsed_time

			else:
			#	self.print('Total Flux:',flux_tot_out,'ff_flux:',None,'simulation_time:',simulation_time,'dpml:',dpml,'res:',resolution,'r:',r,'res_ff:',None , 'source_position:',self.source_position)

				if calculate_source_flux:
					return flux_tot_out, source_flux_out, None, None, elapsed_time
				else:
					return flux_tot_out, None , None

	###OUTPUT DATA##################################################################


if __name__ == "__main__":
	if (len(sys.argv)) != 2:
		print("Not enough arguments")
		exit(0)
	config = read("db/tmp/tmp.json")
	if config == None:
		print("could not open tmp/tmp.json for simulation")
		exit(0)


