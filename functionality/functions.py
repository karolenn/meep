import math as math
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import shgo
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from random import uniform
from functionality.api import read
from copy import deepcopy
import sys
import time




###FUNCTIONS##########################################################


		 
###Create points using the fibbonacci sphere algorithm
def fibspherepts(r,theta,npts,offset):
	xPts = []
	yPts = []
	zPts = []
	range_npts=int((theta/math.pi)*npts)
	increment = math.pi*(3 - math.sqrt(5))

	for n in range(range_npts):
		zPts.append(r*((n*offset-1)+(offset/2)))
		R = r*math.sqrt(1-pow(zPts[n]/r,2))
		phi = (n % npts)*increment
		xPts.append(R*math.cos(phi))
		yPts.append(R*math.sin(phi))
	return(xPts,yPts,zPts)
	
###create point by using spherical coordinate sampling
def sphericalpts(r,theta,phi,theta_pts,phi_pts):
	#initialize list with (0,0,r) points which we don't want to sample more than once (which happens if theta=0 and phi=(0,..,2pi))
	xPts = [0]
	yPts = [0]
	zPts = [-r]
	#the value of angles to loop through
	theta_angles = np.linspace(0+theta/theta_pts,theta,theta_pts)
	phi_angles = np.linspace(0,phi-phi/phi_pts,phi_pts)
	print(theta_angles)
	print(phi_angles)
	for m in range(theta_pts):
		angleTheta=theta_angles[m]
		for n in range(phi_pts):
			anglePhi=phi_angles[n]
			xPts.append(r*math.sin(angleTheta)*math.cos(anglePhi))
			yPts.append(r*math.sin(angleTheta)*math.sin(anglePhi))
			zPts.append(-r*math.cos(angleTheta))
	#print(xPts)
	#print(yPts)
	#print(zPts)
	return xPts,yPts,zPts

#rotate list assistant for spherical points. We need to rotate list depending on phi and sample points on phi
#rotation integer is = {0,..,5}. 0 no rotation, 5 is 5*60 degrees. 
def rotate_list(list,npts,rotation_integer):
	#save the first element 
	first_element = list[0]
	#take every item except first element since that should correspond to point (0,0,r) that should not be rotated
	tmp =deepcopy(list[1:])
	#npts*2 is assumed to be the number of phi sample points per given theta
	phi_pts = 2*npts
	#scale rotation with respect to how many sample points of phi
	rot_points = int(rotation_integer*phi_pts/6)
	for _ in range(rot_points):
		tmp.append(tmp.pop(0))
	#append first item at beginning
	tmp.insert(0,first_element)
	return tmp

def rotate_list2(list,npts,rotation_integer):
	tot = 2*npts*npts
	tmp = [0]*(tot-1)
	phi_pts = 2*npts
	rot_points = int(rotation_integer*phi_pts/6)
	for i in range(0,len(list),phi_pts):
		listRing = deepcopy(list[i:i+phi_pts])
		for _ in range(rot_points):
			listRing.append(listRing.pop(0))
		tmp[i:i+phi_pts] = listRing
	tmp.insert(0,list[0])
	return tmp

#Change the polarization from completely random in space to polarized parallel to the pyramid wall
#and normalized the polarization
def PolarizeInPlane(source_direction,pyramid_height,pyramid_width):
	ph = pyramid_height
	pw = pyramid_width
	old_source_direction = np.asarray(source_direction)
	
	#calculate the normal of the pyramid wall. (Take three points on the wall and the cross prduct)
	#Points are (pwcos(30)/2,pwsin(30)/2,sz/2-sh) and (pwcos(30)/2,-pwsin(30)/2,sz/2-sh) and (0,0,sz/2-sh-ph)
	normal = np.array([pw*ph*math.sin(math.pi/6),0,math.sin(math.pi/6)*math.cos(math.pi/6)*pw**2])
	normal_norm = np.linalg.norm(normal)
	normal = normal/normal_norm

	#Project the source direction into the pyramid wall
	new_source_dir = old_source_direction - (np.dot(old_source_direction,normal))*normal
	#Norm the new projection
	new_source_dir_norm = np.linalg.norm(new_source_dir)
	new_source_dir = new_source_dir / new_source_dir_norm

	#set it to a tuple from a np array
	source_direction_new = (new_source_dir[0],new_source_dir[1],new_source_dir[2])
	
	return source_direction_new

###Get the cross product of the poynting flux on the semi-sphere surface. Get the radial magnitude and return it to later numerically integrate it 
###ff is a point in the far-field containing E,H field for frequency="nfreq"
###input farfield in terms of list [Ex_1freq,Ex2freq,Exnfreq,Ey1freq,..,Hznfreq] where this corresponds to ff at one point
def myPoyntingFlux(ff,nfreq):
	i=nfreq
	P=0 #The poynting flux at point (xPts,yPts,zPts)
	"Calculate the poynting vector in x,y,z direction"
	Px=(ff[i+1]*np.conjugate(ff[i+5])-ff[i+2]*np.conjugate(ff[i+4]))
	Px=Px.real
	Py=(ff[i+2]*np.conjugate(ff[i+3])-ff[i]*np.conjugate(ff[i+5]))
	Py=Py.real
	Pz=(ff[i]*np.conjugate(ff[i+4])-ff[i+1]*np.conjugate(ff[i+3]))
	Pz=Pz.real
	"obtain the radial component of the poynting flux by sqrt(px^2+py^2+pz^2)"
	Pr=math.sqrt(math.pow(Px,2)+math.pow(Py,2)+math.pow(Pz,2))

	return(Pr)


#given a list of far field values, return the poynting flux in radial direction
#[[Ex1_f1,Ey1_f1,..,Hz1_f1,..,Hz1_fn],..,[Exn_f1,..,Hzn_fn]] -> [[Pr1_f1,..,Pr1_fn],..,[Prn_f1,..,Prn_fn]]
def calculate_poynting_values(ff_values):
	npts = len(ff_values)
	#ff_values[0] has nfreq*E,H components = nfreq*6 (ex,ey,ez,hx,hy,hz)
	nfreq = int(len(ff_values[0])/6)
	print(npts,nfreq)
	radial_flux = []
	for n in range(npts):
		i = 0
		radial_flux_tmp = []
		ff_value = ff_values[n]
		for k in range(nfreq):
			Pr = myPoyntingFlux(ff_value,i)
			radial_flux_tmp.append(Pr)
			i = i + 6
		radial_flux.append(radial_flux_tmp)



	return radial_flux

#given a list of poynting flux in radial direction, return a list of the flux for a given freq#
#[[Pr1_f1,..,Pr1_fn],..,[Prn_f1,..,Prn_fn]] -> [Pr1_f1,Pr2_f1,..,Prn_f1]
def get_poynting_per_frequency(radial_flux, freq):
	npts = len(radial_flux)
	tmp = [0]*npts
	for n in range(npts):
		tmp[n] = radial_flux[n][freq]
	return tmp

#integrate (or Riemann sum actually) a list of poynting vectors to get the flux
#[Pr1_f1,...,Prm_f1] -> flux_f1
def get_flux(poyntingvalues, ff_angle, npts, radius):
	surface_element = 2*math.pi*pow(radius,2)*(1-math.cos(ff_angle))/npts
	flux = 0
	for n in range(len(poyntingvalues)):
		flux += poyntingvalues[n]*surface_element
	return flux

#get the cumulative flux per angle
def get_cum_flux_per_angle(Pr_array_freq, total_angle, npts, radius):

	theta_pts = int(math.sqrt((npts-1)/2))
	theta = math.pi / total_angle
	theta_angles = np.linspace(0+theta/theta_pts,theta,theta_pts)
	phi_pts = theta_pts*2
	phi = math.pi*2
	phi_angles = np.linspace(0,phi-phi/phi_pts,phi_pts)
	nr_samples_per_ring = len(phi_angles)
	nr_rings = len(theta_angles)



	cum_flux_per_angle = []

	for n in range(nr_rings):
		angle = theta_angles[n]
		samples = (n+1)*nr_samples_per_ring + 1
		npts = samples
		tmp = get_flux(Pr_array_freq[0:samples],angle,npts,radius)
		cum_flux_per_angle.append(tmp)

	return cum_flux_per_angle

#get the flux per angle given cumulative flux per angle
def get_flux_per_angle(cum_flux_per_angle):
	len_list = len(cum_flux_per_angle)
	flux_per_angle = []
	flux_per_angle.append(cum_flux_per_angle[0])
	for n in range(1,len_list): 
		tmp = cum_flux_per_angle[n] - flux_per_angle[n-1]
		flux_per_angle.append(tmp)
	return flux_per_angle


###Draw a point in 3d space from a uniform distribution
#renamed from get_next to draw_uniform
def draw_uniform(limit_x, limit_y, limit_z):
    x = uniform(limit_x["from"], limit_x["to"])
    y = uniform(limit_y["from"], limit_y["to"])
    z = uniform(limit_z["from"], limit_z["to"])
    return x, y, z

###valid and dist_to_other_pt very similiar
def valid(x,y,z, selected, radius):
    dis = lambda f,s: (f-s)**2
    #print(selected)
    for ptx, pty,ptz in selected:
        if (dis(ptx,x)+ dis(pty,y)+dis(ptz,z))**(1/2) < radius:
            return False
    return True

#check distance between 2 pts, x,y and other_pt = [a,b], yeah I know... Refactoring needed.
def dist_to_other_pt(x,y,z,X,Y,Z):
	dis = lambda f,s: (f-s)**2
	distance = math.sqrt(dis(x,X)+dis(y,Y)+dis(z,Z))
	return distance


###Calculate duplicates in a list of tuples. 
def count(listOfTuple): 
      
    flag = False
  
    # To append Duplicate elements in list 
    coll_list = []   
    coll_cnt = 0
    for t in listOfTuple: 
          
        # To check if Duplicate exist 
        if t in coll_list:   
            flag = True
            continue
          
        else: 
            coll_cnt = 0
            for b in listOfTuple: 
                if b[0] == t[0] and b[1] == t[1]: 
                    coll_cnt = coll_cnt + 1
              
            # To print count if Duplicate of element exist 
            if(coll_cnt > 1):  
                print(t, "-", coll_cnt) 
            coll_list.append(t) 
             
    if flag == False: 
        return "No Duplicates"

def generate_rand_pt(limit_x, limit_y, limit_z, radius, satisfied, max_time, already_selected):
	#check if we already have some data points from simulations
	choice=[]
	if limit_z["from"] == limit_z["to"]:
		z=[]
		for n in range(len(already_selected[0])):
			z.append(limit_z["from"])
		datazipped = zip(already_selected[0],already_selected[1],z)
		selected = list(datazipped)
	else:
		datazipped = zip(already_selected[0],already_selected[1],already_selected[2])
		selected = list(datazipped)
	t = time.time()
	while time.time() - t < max_time:
		if len(choice) >= satisfied:
			print('selected point in generate_rand_pt',selected)
			print('choice point in generate_rand_pt',choice)
			break
		x, y, z = draw_uniform(limit_x, limit_y, limit_z)
		if valid(x,y,z, selected, radius):
			choice.append((x, y, z))
	return choice

#Select the best optima from the optimizers
def RBF_max_point_selector(rbf_optima):
	current_best = {'OPT': 'empty','Function value': -100000, 'sim point':[0,0,0]}
	print(rbf_optima[2])
	for opt in range(len(rbf_optima)):
		print('curr best func',current_best['Function value'])
		print(rbf_optima[opt])
		if (-1*rbf_optima[opt].fun > current_best['Function value']):
				current_best['OPT']="DE" if opt == 0 else "SHGO" if opt == 1 else "DA"
				current_best['Function value']=-1*rbf_optima[opt].fun
				current_best['sim point']=rbf_optima[opt].x.tolist() #convert to list for saving to JSON result
	return current_best

#Choose if we're going for exploration or explotation phase
def Phase_Selector(data,results,rbf_func_max):
	if rbf_func_max - max(results) > 0.02:
		return ('exploit')
	else:
		return ('explore')

#If exploration is chosen, how to select point?
def Exploration_point_selector(data,num_dim):
	pt = Radial_random_chooser(data,num_dim)
	return pt

#Choose next sim point by the radial random chooser
def Radial_random_chooser(data,num_dim):
#	radius=0.5 #radius for the chooser
#	max_time = 10 #max time to run in seconds, else returns ....?? else shrink radius?
#	if False:
#		minx=min(data[0])
#		maxx=max(data[0])
#		miny=min(data[1])
#		maxy=max(data[1])
#		minz=min(data[2])
#		maxz=max(data[2])
#		template = read("db/tmp/tmp.json")
#		limit_x = {"from": minx, "to":maxx }
#		limit_y = {"from": miny, "to":maxy }
#		limit_z = {"from": minz, "to": maxz }
#		exploration_pt = generate_rand_pt(limit_x, limit_y, limit_z, radius, 1, max_time, data)
#		#if fail to find point, shrink radius to half
#		while exploration_pt == []:
#			exploration_pt = generate_rand_pt(limit_x, limit_y, limit_z, radius*0.5, 1, max_time, data)
#		return exploration_pt
###Two methods to choose next random point in variable space. Minmax or "shrink method"###
	Choice = 'NOT shrinker'
	if Choice == 'shrinker':
		exploration_pt=rand_pt_shrink(data)
	else:
		exploration_pt=rand_pt_minmax(data)

	return exploration_pt

def rand_pt_shrink(data):
	minx=min(data[0]) #min pyramid width
	maxx=max(data[0]) #max pyramid width
	miny=min(data[1]) #min source pos
	maxy=max(data[1]) #max source pos
	template = read("db/tmp/tmp.json")
   # for i, name in enumerate(args):
   #     template["pyramid"][name] = result[i]
	z = template["pyramid"]["source_position"]
	limit_x = {"from": minx, "to":maxx }
	limit_y = {"from": miny, "to":maxy }
	limit_z = {"from": z, "to": z }
	exploration_pt = generate_rand_pt(limit_x, limit_y, limit_z, radius, 1, max_time, data)
	#if fail to find point, shrink radius to half
	while exploration_pt == []:
		radius=radius*0.5
		exploration_pt = generate_rand_pt(limit_x, limit_y, limit_z, radius, 1, max_time, data)
		print('im looking for a new point')
		print('radius here',radius)
	print('radius',radius)
	exploration_pt=list(exploration_pt[0])
	exploration_pt[2]=exploration_pt[0]*math.tan(math.pi*62/180)/2
	#exploration point is a tuple within a list [(1,2)] so it is converted to a list
	return exploration_pt

#select "best" (pt furthest away from others) pt if pt is not found
def rand_pt_minmax(data):
	if len(data)==2:
		satisfied = 1000 # number of random points to generate
	#data[0]=data[0][:20]
	#data[1]=data[1][:20]
		minx=min(data[0]) #min pyramid width
		maxx=max(data[0]) #max pyramid width
		miny=min(data[1]) #min source pos
		maxy=max(data[1]) #max source pos
		#convert to 1x1 size plane
		dict01={"from": 0, "to":1}
		dict1={"from": minx, "to":maxx}
		dict2={"from": miny, "to":maxy}
		minX = 0
		maxX = 1
		minY = 0
		maxY = 1
		#check if we already have some data points from simulations
		limit_x = {"from": minx, "to":maxx }
		limit_y = {"from": miny, "to":maxy }
		list_rand_pts=[]
		for i in range(satisfied):
			randX, randY, randZ = draw_uniform(dict01,dict01,dict01)
			list_rand_pts.append((randX,randY))
		dist_to_closest_pt = 990
		current_best_pt = [0,0]
		current_longest_dist = 0
		#	print('length of data',len(data[0]))
		#	print('all pts',list_rand_pts)
		for k in range(satisfied): #we generate #satisfied number of random pts that we loop through
			randX = list_rand_pts[k][0] #current randomised pt position X, Y below
			randY = list_rand_pts[k][1]
		#	print('***--------------------------***')
		#	print('Random pt being tested:',randX,randY)
			for n in range(len(data[0])): #We compare the distance between randXY pt with data pts
				dX=(data[0][n]-minx)/(maxx-minx) #convert distance to [0,1] distance instead of [a,b]
				dY=(data[1][n]-miny)/(maxy-miny)
			#	dX=data[0][n]
			#	dY=data[1][n]
			#	print('Current data pt being compared against:',dX,dY)
				current_dist = dist_to_other_pt(randX,randY,0,dX,dY,0) #check euclidian distance between rand pt and pt in data set
			#	print('current distance to data pt from rand pt',current_dist)
			#	print('dist to closest pt currently',dist_to_closest_pt)
				if current_dist < dist_to_closest_pt: #min part of maxmin, we need to save closest, i.e worst distance
					dist_to_closest_pt = current_dist
					#print('distance to closest pt',round(current_dist,3),'data pt',round(dX,3),round(dY,3),'rand pt',round(randX,3),round(randY,3))
			if dist_to_closest_pt > current_longest_dist: #max part of minmax
				current_best_pt=[randX,randY]
				current_longest_dist = dist_to_closest_pt
			#	print(dX,dY)
			#	print('NEW BEST POINT!')
			#	print('current best',current_best_pt,'distance:',current_dist,'clo',current_longest_dist)
			#	print('--------------')
			current_dist = 990 #reset current distance
			dist_to_closest_pt=999
		current_best_pt[0]=current_best_pt[0]*(maxx-minx)+minx
		current_best_pt[1]=current_best_pt[1]*(maxy-miny)+miny
	#	print('returning pt',current_best_pt)
		return current_best_pt
	elif len(data)==3:
		satisfied = 5000 # number of random points to generate
		minx=min(data[0])
		maxx=max(data[0])
		miny=min(data[1])
		maxy=max(data[1]) 
		minz=min(data[2])
		maxz=max(data[2])
		#convert to 1x1 size volume
		dict01={"from": 0, "to":1}
		dict1={"from": minx, "to":maxx}
		dict2={"from": miny, "to":maxy}
		dict3={"from": minz, "to":maxz}
		minX = 0
		maxX = 1
		minY = 0
		maxY = 1
		#check if we already have some data points from simulations
		limit_x = {"from": minx, "to":maxx }
		limit_y = {"from": miny, "to":maxy }
		limit_z = {"from": minz, "to":maxz }
		list_rand_pts=[]
		for i in range(satisfied):
			randX, randY, randZ = get_next(dict01,dict01,dict01)
			list_rand_pts.append((randX,randY,randZ))
		dist_to_closest_pt = 990
		current_best_pt = [0,0,0]
		current_longest_dist = 0
		for k in range(satisfied): #we generate #satisfied number of random pts that we loop through
			randX = list_rand_pts[k][0] #current randomised pt position X, Y below
			randY = list_rand_pts[k][1]
			randZ = list_rand_pts[k][2]
			for n in range(len(data[0])): #We compare the distance between randXY pt with data pts
				dX=(data[0][n]-minx)/(maxx-minx) #convert distance to [0,1] distance instead of [a,b]
				dY=(data[1][n]-miny)/(maxy-miny)
				dZ=(data[2][n]-minz)/(maxz-minz)
				current_dist = dist_to_other_pt(randX,randY,randZ,dX,dY,dZ) #check euclidian distance between rand pt and pt in data set
				if current_dist < dist_to_closest_pt: #min part of maxmin, we need to save closest, i.e worst distance
					dist_to_closest_pt = current_dist
			if dist_to_closest_pt > current_longest_dist: #max part of minmax
				current_best_pt=[randX,randY,randZ]
				current_longest_dist = dist_to_closest_pt
			current_dist = 990 #reset current distance
			dist_to_closest_pt=999
		current_best_pt[0]=current_best_pt[0]*(maxx-minx)+minx
		current_best_pt[1]=current_best_pt[1]*(maxy-miny)+miny
		current_best_pt[2]=current_best_pt[2]*(maxz-minz)+minz
		return current_best_pt
	else:
		raise ValueError ('minmax rand point gets wrong data dim!')







def utility_function(data,results,ff_calc):
	#first we need to 'unpack' data and results into arrays
	num_dim=len(data)
	if not (2 <= num_dim <= 3):
		raise ValueError (' utility function for D!=3 OR 2 not completed')
	elif num_dim == 2:
		merge=list(zip(data[0],data[1]))
		count(merge)
		if count(merge) != 'No Duplicates':
			raise ValueError ('Duplicate simulation results in database. RBF can can not handle duplicate data points')
		#need to pick out above/under & certain frequency for result
		##Check above/under for results. Pick the center frequency

		RBF_Func=Rbf(data[0],data[1],results,function='thin_plate')
		minx=min(data[0])
		maxx=max(data[0])
		miny=min(data[1])
		maxy=max(data[1])
		minf=min(results)
		maxf=max(results)
		index_max = np.argmax(results)
		#print(minx,maxx,miny,maxy,minf,maxf)
		x = np.linspace(minx,maxx,num=30)
		y = np.linspace(miny,maxy,num=30)
		f = np.linspace(minf,maxf,num=30)
		#optimizer
		init_guess = np.array([(maxx-minx)/2,(maxy-miny)/2])
		init_guess=np.array([data[0][index_max],data[1][index_max]])
		rbf_optima=[]
									
		#global minimizers
		def RBF_TO_OPT(x):
			return (-1*RBF_Func(x[0],x[1]))
		rbf_optima.append(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		rbf_optima.append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		rbf_optima.append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		#if true, exploit, if false, explore
		current_best_rbf = RBF_max_point_selector(rbf_optima)
		exploit_pt = current_best_rbf['sim point']
		explore_pt = Exploration_point_selector(data,num_dim)
		print('exploit',exploit_pt,'explore',explore_pt)
		X,Y=np.meshgrid(x,y)
		z = RBF_Func(X,Y)
		if False:
			# FOR TESTING PURPOSES
			fig = plt.figure()
			ax = plt.axes(projection='3d')
			ax.text(explore_pt[0],explore_pt[1],-1*RBF_TO_OPT(explore_pt),'EXPLORE POINT',fontsize=12)
			ax.text(exploit_pt[0],exploit_pt[1],-1*RBF_TO_OPT(exploit_pt),'EXPLOIT POINT',fontsize=12)
			ax.scatter(exploit_pt[0],exploit_pt[1],-1*RBF_TO_OPT(exploit_pt),s=50,c='r',marker='D')
			ax.scatter(explore_pt[0],explore_pt[1],-1*RBF_TO_OPT(explore_pt),s=50,c='g',marker='D')
			ax.scatter(data[0],data[1],results,alpha=1,c='k')
			for i in range(len(data[0])):
				ax.text(data[0][i],data[0][i],results[i],str(i))
			ax.plot_surface(X,Y,z,cmap=cm.jet)
			ax.set_xlabel('pyramid width')
			ax.set_ylabel('source position')
			plt.show()
		print(current_best_rbf)
		if Phase_Selector(data,results,current_best_rbf['Function value'])=='exploit':
			next_sim = exploit_pt
			print('next_sim',next_sim)
			print('EXPLOATATIONS','point',next_sim)
		else: 
			next_sim = explore_pt
			print('next_sim',next_sim)
			print('EXPLORATION','point',next_sim)
		return next_sim, max(results), Phase_Selector(data,results,current_best_rbf['Function value']),current_best_rbf

	elif num_dim == 3:
		merge=list(zip(data[0],data[1],data[2]))
		count(merge)
		print('data',data)
		print('results',results)
		#need to pick out above/under & certain frequency for result
		##Check above/under for results. Pick the center frequency

		RBF_Func=Rbf(data[0],data[1],data[2],results,function='thin_plate')
		minx=min(data[0])
		maxx=max(data[0])
		miny=min(data[1])
		maxy=max(data[1])
		minz=min(data[2])
		maxz=max(data[2])
		minf=min(results)
		maxf=max(results)
		index_max = np.argmax(results)
		#print(minx,maxx,miny,maxy,minf,maxf)
		x = np.linspace(minx,maxx,num=30)
		y = np.linspace(miny,maxy,num=30)
		z = np.linspace(minz,maxz,num=30)
		f = np.linspace(minf,maxf,num=30)
		#optimizer
		init_guess = np.array([(maxx-minx)/2,(maxy-miny)/2,(maxz-minz)/2])
		init_guess=np.array([data[0][index_max],data[1][index_max],data[2][index_max]])
		rbf_optima=[]
									
		#global minimizers
		def RBF_TO_OPT(x):
			return (-1*RBF_Func(x[0],x[1],x[2]))
		rbf_optima.append(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		#if true, exploit, if false, explore
		current_best_rbf = RBF_max_point_selector(rbf_optima)
		exploit_pt = current_best_rbf['sim point']
		explore_pt = Exploration_point_selector(data,num_dim)
		print('exploit',exploit_pt,'explore',explore_pt)
		X,Y,Z=np.meshgrid(x,y,z)
		z = RBF_Func(X,Y,X)
		if False:
			# FOR TESTING PURPOSES
			fig = plt.figure()
			ax = plt.axes(projection='3d')
			ax.text(explore_pt[0],explore_pt[1],-1*RBF_TO_OPT(explore_pt),'EXPLORE POINT',fontsize=12)
			ax.text(exploit_pt[0],exploit_pt[1],-1*RBF_TO_OPT(exploit_pt),'EXPLOIT POINT',fontsize=12)
			ax.scatter(exploit_pt[0],exploit_pt[1],-1*RBF_TO_OPT(exploit_pt),s=50,c='r',marker='D')
			ax.scatter(explore_pt[0],explore_pt[1],-1*RBF_TO_OPT(explore_pt),s=50,c='g',marker='D')
			ax.scatter(data[0],data[1],results,alpha=1,c='k')
			for i in range(len(data[0])):
				ax.text(data[0][i],data[0][i],results[i],str(i))
			ax.plot_surface(X,Y,z,cmap=cm.jet)
			ax.set_xlabel('pyramid width')
			ax.set_ylabel('source position')
			plt.show()
		print(current_best_rbf)
		if Phase_Selector(data,results,current_best_rbf['Function value'])=='exploit':
			next_sim = exploit_pt
			print('next_sim',next_sim)
			print('EXPLOATATIONS','point',next_sim)
		else: 
			next_sim = explore_pt
			print('next_sim',next_sim)
			print('EXPLORATION','point',next_sim)
		return next_sim, max(results), Phase_Selector(data,results,current_best_rbf['Function value']),current_best_rbf






	
	

	
	


	
	
