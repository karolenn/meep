import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import math as math
import time
#import functions
#from functions import CalculatePowerRatio 			#import function that calculates ratio of total and angle out

###PARAMETERS FOR THE SIMULATION##############################################

#from mpl_toolkits.mplot3d import Axes3D


if (len(sys.argv)) != 6:
    print("Not enough arguments")
    exit(0)

"1 unit distance is meep is thought of as 1 micrometer here."

"Structure Geometry"
pyramid_height=float(sys.argv[4])				#height of the pyramid in meep units 3.2
pyramid_width=float(2.4)					#width measured from edge to edge 2.6
substrate_height=pyramid_height/20				#height of the substrate, measured as fraction of pyramid height
"Cell size"
sx=pyramid_width*(6/5)						#size of the cell in xy-plane is measured as a fraction of pyramid width
sy=sx
sz=pyramid_height*(6/5)						#z-"height" of sim. cell measured as a fraction of pyramid height.		

					
dpml=0.1 							#thickness of pml layers
padding=0.1							##distance from pml_layers to flux regions so PML don't overlap flux regions

"Inarguments for the simulation"
resolution= int(sys.argv[1])					#resolution of the pyramid. Measured as number of pixels / unit distance
simulation_time=int(sys.argv[2])				#simulation time for the sim. #Multiply by a and divide by c to get time in fs.
source_pos=float(sys.argv[3])					#pos of source measured	measured as fraction of tot. pyramid height from top. 
cell=mp.Vector3(sx+2*dpml,sy+2*dpml,sz+2*dpml)	 		#size of the simulation cell in meep units
						

"Direction for source"
source_direction=mp.Ey						#polasation of the gaussian source. 
								#symmetry has normal in x & y-direction, phase-even z-dir 


"Frequency parameters for gaussian source"
fcen=2								#center frequency
df=0.4	   							#frequency span, ranging from 1.7-2.3 
nfreq=1								#number of frequencies sampled


"Calculation and plotting parameters"
use_symmetries = 'true'						#run sim with 4-fold symmetries? reduces calc time by ~1/4th
calculate_flux = 'true'
plot_epsilon = 'false'						#Plot epsilon data in xy,yz,xz slices
angle = math.pi/6						#calculate flux at angle measured from top of the pyramid
far_field_calculation = 'true'					#calculate far fields
mp_ff_calculation = 'false'					#calculate far fields using meeps function
								

###GEOMETRY FOR THE SIMULATION#################################################

"Material parameters"
GaN = mp.Medium(epsilon=5.76)					#GaN n^2=epsilon, n=~2.4 
air = mp.Medium(epsilon=1)					#air dielectric value
SubstrateEps = mp.Medium(epsilon=5.76)				#substrate epsilon


"Pyramid variables, h is length from origo to parallel edge, v is length from center of edge to first corner."
h=pyramid_width/2
v=h
sh=substrate_height

"Function for creating pyramid"

"isInsidexy checks if a point is inside the pyramid or not. If true, sets epsilon to 5.76, if false, sets it to 1. "
"the function loops over the height of the pyramid in z-direction. It transform the test point to vertice 2. It then performs 3 test to check if the point"
"is inside a hexagonal area or not. Decreasing/increasing h over the span of z then forms a pyramid. Courtesy to playchilla.com for logic test if a point is inside a hexagon or not."

"loops over the height of the pyramid in z-direction. "
def isInsidexy(vec):
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

"Not working. Alternative of drawing the pyramid by using constant planes to check material."
def isInside(vec):
	m=(3+math.sqrt(3))/2
	k=-m
	v=pyramid_width/(2*pyramid_height)*vec.z-(pyramid_width/(2*pyramid_height))*(sz/2-sh-pyramid_height)

	while (vec.z <= sz/2-sh-1 and vec.z >= sz/2-sh-pyramid_height):
		if (vec.y <= h*math.cos(math.pi/6) and vec.y >= 0 and vec.x <= (vec.y-m)/k):
			return GaN
		else:
			return air

	else:
		return air

"Geometry to define the Substrate"

Substrate = []


Substrate=[mp.Block(center=mp.Vector3(0,0,sz/2-sh/2+dpml/2),
			size=mp.Vector3(sx+2*dpml,sy+2*dpml,sh+dpml),
			material=SubstrateEps)]



###FUNCTIONS##########################################################
"Functions for the simulation. TODO: Move to another folder."
def fibspherepts(r,theta,npts):
	startTime=time.time()
	global xPts
	xPts=[]

	global yPts
	yPts=[]

	global zPts
	zPts=[]
	

	offset=2/npts
	range_npts=int((theta/math.pi)*npts)
	increment = math.pi*(3 - math.sqrt(5))

	for n in range(npts):
		yPts.append(r*((n*offset-1)+(offset/2)))
		R = r*math.sqrt(1-pow(yPts[n]/r,2))
		phi = (n % npts)*increment
		xPts.append(R*math.cos(phi))
		zPts.append(R*math.sin(phi))
	
	print('time taken:',time.time()-startTime)

def spherepts(r,theta,phi,npts):
	global xPts
	xPts=[]

	global yPts
	yPts=[]

	global zPts
	zPts=[]

	for n in range(npts):
		angleN=phi*(n/npts)

		for m in range(npts):
			angleM=theta*(m/npts)

			xPts.append(r*math.sin(angleM)*math.cos(angleN))
			yPts.append(-r*math.sin(angleM)*math.sin(angleN))
			zPts.append(-r*math.cos(angleM))

def planepts(r,nptsline,theta):

	d=r*math.tan(theta/2)

	global xPts
	xPts=[]

	global yPts
	yPts=[]

	global zPts
	zPts=[]

	delta=2*d/nptsline
	
	for k in range(nptsline):
		zPts.append(-r)	

	for n in range(nptsline):
		xPts.append(-d+delta*n)	

		for m in range(nptsline):
			yPts.append(-d+delta*m)

def output_power(sim):
	print('Total flux:',mp.get_fluxes(flux_total))

		





###SYMMETRIES#########################################################
"Symmetry logic."
if use_symmetries == 'true':

	if source_direction ==	mp.Ex:
		symmetry=[mp.Mirror(mp.Y),mp.Mirror(mp.X,phase=-1)]
	elif source_direction == mp.Ey:
		symmetry=[mp.Mirror(mp.X),mp.Mirror(mp.Y,phase=-1)]
	elif source_direction == mp.Ez:
		symmetry =[mp.Mirror(mp.X),mp.Mirror(mp.Z,phase=-1)]
	else:
		symmetry = []


	print('symmetry',symmetry)

###PML_LAYERS###################################################################

pml_layer=[mp.PML(dpml)]

###SOURCE#######################################################################

"A gaussian with pulse source proportional to exp(-iwt-(t-t_0)^2/(2w^2))"

"Source position"
abs_source_pos=sz/2-sh-pyramid_height+pyramid_height*(source_pos)

source=[mp.Source(mp.GaussianSource(frequency=fcen,fwidth=df,cutoff=2),	#gaussian current-source
		component=source_direction,
		center=mp.Vector3(0,0,abs_source_pos))]

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

"These regions define the borders of the cell with distance 'padding' between the flux region and the dpml region to avoid calculation errors."
if calculate_flux == 'true':
	fluxregion1=mp.FluxRegion(					#region x to calculate flux from
		center=mp.Vector3(sx/2-padding,0,0),
		size=mp.Vector3(0,sy-padding*2,sz-padding*2),
		direction=mp.X)

	fluxregion2=mp.FluxRegion(					# region -x to calculate flux from
		center=mp.Vector3(-sx/2+padding,0,0),
		size=mp.Vector3(0,sy-padding*2,sz-padding*2),
		direction=mp.X,
		weight=-1)

	fluxregion3=mp.FluxRegion(					#region y to calculate flux from
		center=mp.Vector3(0,sy/2-padding,0),
		size=mp.Vector3(sx-padding*2,0,sz-padding*2),
		direction=mp.Y)

	fluxregion4=mp.FluxRegion(					#region -y to calculate flux from
		center=mp.Vector3(0,-sy/2+padding,0),
		size=mp.Vector3(sx-padding*2,0,sz-padding*2),
		direction=mp.Y,
		weight=-1)

	fluxregion5=mp.FluxRegion(					#z-bottom region to calculate flux from
		center=mp.Vector3(0,0,sz/2-padding),
		size=mp.Vector3(sx-padding*2,sy-padding*2,0),
		direction=mp.Z)

	fluxregion6=mp.FluxRegion(					#z-top region to calculate flux from
		center=mp.Vector3(0,0,-sz/2+padding),
		size=mp.Vector3(sx-padding*2,sy-padding*2,0),
		direction=mp.Z,
		weight=-1)

###ANGLE REGION#################################################################

	"This region is the area corresponding to an 'angle' above the pyramid. TODO: Check relationship between this flux and far field flux" 


	#region_size=(sz/2-padding)*math.tan(angle/2)

	#fluxregion_angle=mp.FluxRegion(
	#		center=mp.Vector3(0,0,-sz/2+padding),
	#		size=mp.Vector3(region_size*2,region_size*2,0),
	#		direction=mp.Z,
	#		weight=-1)

###FIELD CALCULATIONS###########################################################

	"Tells meep with the function 'add_flux' to collect and calculate the flux in the corresponding regions and put them in a flux data object"
	
	flux_total=sim.add_flux(fcen,df,nfreq,
		fluxregion1,fluxregion2,fluxregion3,fluxregion4,fluxregion5,fluxregion6)	#calculate flux for flux regions

	#flux_data_tot=sim.get_flux_data(flux_total)					#get flux data for later reloading

###ANGLE FIELD CALCULATIONS#####################################################

#flux_angle=sim.add_flux(fcen,df,nfreq,fluxregion_angle)			#calculate flux for angle region

###FAR FIELD REGION#############################################################

"The simulation calculates the far field flux from the regions 1-5 below. It correspons to the air above and at the side of the pyramids. The edge of the simulation cell that touches the substrate is not added to this region. Far-field calculations can not handle different materials."

nearfieldregion1=mp.Near2FarRegion(
		center=mp.Vector3(sx/2,0,-sh/2),
		size=mp.Vector3(0,sy,sz-sh-padding*2),
		direction=mp.X)

nearfieldregion2=mp.Near2FarRegion(
		center=mp.Vector3(-sx/2,0,-sh/2),
		size=mp.Vector3(0,sy,sz-sh-padding*2),
		direction=mp.X,
		weight=-1)

nearfieldregion3=mp.Near2FarRegion(
		center=mp.Vector3(sy/2,0,-sh/2),
		size=mp.Vector3(sx,0,sz-sh-padding*2),
		direction=mp.Y)

nearfieldregion4=mp.Near2FarRegion(
		center=mp.Vector3(-sy/2,0,-sh/2),
		size=mp.Vector3(sx,0,sz-sh-padding*2),
		direction=mp.Y,
		weight=-1)

nearfieldregion5=mp.Near2FarRegion(					#nearfield -z. above pyramid.		
		center=mp.Vector3(0,0,-sz/2+padding),
		size=mp.Vector3(sx-padding*2,sy-padding*2,0),
		direction=mp.Z,
		weight=-1)


nearfield=sim.add_near2far(fcen,df,nfreq,nearfieldregion1,nearfieldregion2,nearfieldregion3,nearfieldregion4,nearfieldregion5)



###RUN##########################################################################
"The run constructor for meep."
#sim.run(until_after_sources=mp.stop_when_fields_decayed(2, source_direction, mp.Vector3(0,0,abs_source_pos+0.2), 1e-3))
sim.run(
mp.at_beginning(mp.output_epsilon),
#mp.at_every(2,output_power),
until=simulation_time)
#sim.run(until_after_sources=mp.stop_when_fields_decayed(10,mp.Ez,mp.Vector3(0,0.5,0),1e-2))


###OUTPUT CALCULATIONS##########################################################

"Calculate the poynting flux given the far field values of E, H."

fibsphere = 'true'
planeff = 'false'
r=(1/fcen)*50
if far_field_calculation == 'true':
	P_tot_ff = np.zeros(nfreq)
							#distance to far-field
	npts=1600							#number of far-field points
	Px=0
	Py=0
	Pz=0
	theta=math.pi/6
	phi=math.pi*2

	if fibsphere == 'true':
		fibspherepts(r,theta,npts)

	elif planeff == 'true':
		npts=int(round(math.sqrt(npts)))
		planepts(r,npts,theta)
		print('x:', xPts)
		print('y:', yPts)
		print('z:', zPts)

	else: 
		spherepts(r,theta,phi,round(math.sqrt(npts)))
		npts=int(math.pow(round(math.sqrt(npts)),2))

	for n in range(npts):


		ff=sim.get_farfield(nearfield, mp.Vector3(xPts[n],yPts[n],zPts[n]))
		"P_tot_ff calculated as (1/2)Re(E x H*)scalar-product(sin(angleM)cos(angleN),sin(angleM)sin(angleN),-cos(angleM)) where H* is the complex conjugate and this corresponds to the average power density in radial direction from the simulation cell at a distance of r micrometers."

		i=0
		for k in range(nfreq):
			Px=(ff[i+1]*np.conjugate(ff[i+5])-ff[i+2]*np.conjugate(ff[i+4]))
			Px=Px.real
			Py=(ff[i+2]*np.conjugate(ff[i+3])-ff[i]*np.conjugate(ff[i+5]))
			Py=Py.real
			Pz=(ff[i]*np.conjugate(ff[i+4])-ff[i+1]*np.conjugate(ff[i+3]))
			Pz=Pz.real
			Pr=math.sqrt(math.pow(Px,2)+math.pow(Py,2)+math.pow(Pz,2))					
			surface_Element=2*math.pi*pow(r,2)*(1-math.cos(theta))/npts
			P_tot_ff[k] += surface_Element*(1)*np.real(Pr)

			i=i+6


			"P_tot_ff[k] is calculated for each freq. Now the loop should make the spacing between two points larger the further down the sphere we goes and surface_elements might overlap here. Check in the future the errors."


	##CALCULATE FLUX OUT FROM BOX###########################################
if calculate_flux == 'true':
	"Initialize variables to be used, pf stands for 'per frequency'"
	flux_tot_value = np.zeros(nfreq)						#total flux out from box
	flux_tot_ff_ratio=np.zeros(nfreq)						
	flux_tot_pf = []
	flux_tot_pf_ratio = []

	flux_tot_out = mp.get_fluxes(flux_total)			#save total flux data

	if mp_ff_calculation == 'true':
		volr=mp.Volume(center=mp.Vector3(0,0,-r),size=mp.Vector3(2*r*math.tan(angle/2),2*r*math.tan(angle/2),0),dims=3)
		flux_tot_freqs = mp.get_flux_freqs(flux_total)			#save flux frequencies
		res_ff=0.3
		ff_flux = -nearfield.flux(mp.Z,volr,res_ff)[0]

		print('ff_flux',ff_flux)
	
	#print('Total Flux:',flux_tot_out,'time:',simulation_time,'dpml:',dpml,'res:',resolution,'offset:',big_offset)

	if far_field_calculation == 'true':
		"the for loop sums up the flux for all frequencies and stores it in flux_tot_value and flux_top_value"
		"it also calculates the ratio betwewen top and total flux out for all frequencies and stores it in an array 'flux_tot_pf_ratio'"

		for i in range(nfreq):

			flux_tot_ff_ratio[i] =P_tot_ff[i]/flux_tot_out[i]			#sums up the total flux out
		print('flux_tot_ff_ratio',flux_tot_ff_ratio)

		print('Total Flux:',flux_tot_out,'Flux farfield:',P_tot_ff,'ratio:',flux_tot_ff_ratio
#,'ff_flux:',ff_flux
,'simulation_time:',simulation_time,'dpml:',dpml,'res:',resolution,'r:',r,'npts:',npts,'source_pos:',source_pos,'pyramid_height:',pyramid_height,
'pyramid_width:',pyramid_width)

	else:
		print('Total Flux:',flux_tot_out,'ff_flux:',ff_flux,'simulation_time:',simulation_time,'dpml:',dpml,'res:',resolution,'r:',r,'res_ff:',res_ff, 'source_pos:',source_pos)




	##CALCULATES FLUX OUT FROM BOX AT AN ANGLE#############################
"This function does the same as above but the calculations area for the area above the pyramid corresponding to 'angle'"
"THIS IS DEFINED IN A FUNCTION, CALCULATEPOWERATIO INSTEAD!!!!"
	#if calculate_at_an_angle == 'true':

	#	flux_angle_value = 0
	#	flux_angle_pf = []
	#	flux_tot_pf_angleratio = []

	#	for i in range(nfreq):

	#		flux_angle_value +=flux_angle_out[i]		#sums up the total flux out at an angle

	#		flux_tot_pf_angleratio.append(flux_angle_out[i]/flux_tot_out[i])	#takes ratio flux angle/total for each freq

	#	ratioAngle=flux_angle_value/flux_tot_value		#takes the ratio for all flux, same as above
	#	print("Flux at an angle / Total flux:",ratioAngle)



###OUTPUT DATA##################################################################

"Plot convergence data"

"Some plotting. Do not use if you run the script and iterate over many simulations."



plot_all_below = 'true'

if plot_all_below == 'false':



	"Collect value and plot Ey for x=0 plane"
	ez_yzdata = sim.get_array(center=mp.Vector3(0,0,0), size=mp.Vector3(0,sy,sz), component=mp.Ey)
	plt.figure(dpi=100)
	#plt.imshow(eps_datayz.transpose(), interpolation='spline36', cmap='binary')
	plt.imshow(ez_yzdata.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
	plt.axis('off')
	plt.show()



	if plot_epsilon == 'true':


		"Collect value of epsilon for x=0,y=0,z=0 planes"
		#eps_dataxy = sim.get_array(center=mp.Vector3(0,0,sz/2-sh-padding), size=mp.Vector3(sx,sy,0), component=mp.Dielectric)
		eps_dataxy = sim.get_array(center=mp.Vector3(0,0,0), size=mp.Vector3(sx,sy,0), component=mp.Dielectric)
		eps_datayz = sim.get_array(center=mp.Vector3(0,0,0), size=mp.Vector3(0,sy,sz), component=mp.Dielectric)
		eps_dataxz = sim.get_array(center=mp.Vector3(0,0,0), size=mp.Vector3(sx,0,sz), component=mp.Dielectric)


		plt.figure(dpi=100)
		plt.imshow(eps_dataxy.transpose(), interpolation='spline36',  cmap='binary')
		plt.axis('on')
		plt.ylabel('y')
		plt.xlabel('x')
		plt.show()

		plt.figure(dpi=100)
		plt.imshow(eps_datayz.transpose(), interpolation='spline36',  cmap='binary')
		plt.axis('on')
		plt.ylabel('z')
		plt.xlabel('y')
		plt.show()

		plt.figure(dpi=100)
		plt.imshow(eps_dataxz.transpose(), interpolation='spline36',  cmap='binary')
		plt.axis('on')
		plt.ylabel('z')
		plt.xlabel('x')
		plt.show()






	if calculate_flux == 'true NOOOO':

		"These two plots plot the ratio between 'flux top' or 'flux angle' over the 'total flux out' where the x-axis are meep frequencies."
		plt.plot(flux_tot_freqs,flux_tot_pf_angleratio)
		plt.plot(flux_tot_freqs,flux_tot_pf_ratio)
		leg = plt.legend(loc='upper center', ncol=2, mode="expand", bbox_to_anchor=(0, 1.06, 1, .102),fontsize=15)
		leg.get_frame().set_alpha(0.5)
		plt.xlabel('frequency')
		plt.show()

		"We'd like to plot over wavelengths. So we convert the frequencies to wavelengths in nanometer. The transformation is not linear (?) unfortunately."

		flux_tot_lambda=[]

		for i in range(nfreq):
			flux_tot_lambda.append(1/flux_tot_freqs[i])
			flux_tot_lambda[i]=flux_tot_lambda[i]*1000
		
		"For testing purposes."
		print("P_top/P_out:",flux_tot_pf_ratio)
		print("P_angle/P_out:",flux_tot_pf_angleratio)
		print("flux_tot_lambda:",flux_tot_lambda)


		"These two plots plot the ratio between 'flux top' and 'flux angle' over 'total flux out' where x-axis are wavelengths in nanometer."
		plt.plot(flux_tot_lambda,flux_tot_pf_angleratio,label=r'$\frac{P_{angle}}{P_{total}}$ ')
		plt.plot(flux_tot_lambda,flux_tot_pf_ratio,label=r'$\frac{P_{top}}{P_{total}}$')
		leg = plt.legend(loc='upper center', ncol=2, mode="expand", bbox_to_anchor=(0, 1.06, 1, .102),fontsize=15)
		leg.get_frame().set_alpha(0.5)
		plt.xlabel(r'$\lambda$ [nm]')
		plt.show()

		"These two plots plot total flux out and top flux out where the x-axis are meep frequencies."
		plt.plot(flux_tot_freqs,flux_tot_out,label='total flux out')
		plt.plot(flux_tot_freqs,flux_top_out,label='flux out from top')
		if calculate_at_an_angle == 'true':
			plt.plot(flux_tot_freqs,flux_angle_out,label='flux out at angle')
		leg = plt.legend(loc='upper center', ncol=2, mode="expand", bbox_to_anchor=(0, 1.06, 1, .102))
		leg.get_frame().set_alpha(0.5)
		plt.xlabel('frequency')
		plt.show()






