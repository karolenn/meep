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
from .api import read
import sys
import time
#from skimage import measure


###FUNCTIONS##########################################################
def fibspherepts(r,theta,npts,xPts,yPts,zPts,offset):
	range_npts=int((theta/math.pi)*npts)
	increment = math.pi*(3 - math.sqrt(5))

	for n in range(range_npts):
		zPts.append(r*((n*offset-1)+(offset/2)))
		R = r*math.sqrt(1-pow(zPts[n]/r,2))
		phi = (n % npts)*increment
		xPts.append(R*math.cos(phi))
		yPts.append(R*math.sin(phi))
	return(xPts,yPts,zPts,npts)
	

def sphericalpts(r,theta,phi,npts,xPts,yPts,zPts):

	for n in range(npts):
		angleTheta=(n/npts)*theta
		for m in range(npts):
			anglePhi=(m/npts)*phi
			xPts.append(r*math.sin(angleTheta)*math.cos(anglePhi))
			yPts.append(r*math.sin(angleTheta)*math.sin(anglePhi))
			zPts.append(-r*math.cos(angleTheta))

	return(xPts,yPts,zPts)

def plot_data(sim_name,x,y):

    db = read("db/initial_results/{}.json".format(sim_name))
    if db == None:
        print("could not open db/initial_results/{}.json".format(sim_name))
        exit(0)
    
#    for config in db:
 #       pyramid = Pyramid(config["pyramid"])
  #      result = pyramid.simulate(config["simulate"])
   #     data = sim_to_json(config, result)
    #    write_result("db/initial_results/{}.json".format(sim_name), data)
    array_data=db_to_array(db)

"Get the cross product of the poynting flux on the semi-sphere surface. Get the radial magnitude and numerically integrate it "
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

def get_next(limit_x, limit_y, limit_z):
    x = uniform(limit_x["from"], limit_x["to"])
    y = uniform(limit_y["from"], limit_y["to"])
    z = uniform(limit_z["from"], limit_z["to"])
    return x, y, z

def valid(x,y,z, selected, radius):
    dis = lambda f,s: (f-s)**2
    #print(selected)
    for ptx, pty,ptz in selected:
        if (dis(ptx,x)+ dis(pty,y)+dis(ptz,z))**(1/2) < radius:
            return False
    return True
  #  limit_x = {"from": 1, "to":3 }
  #  limit_y = {"from": 1, "to":3 }
  # limit_z  z = {"from": 0.06, "to": 0.06 }


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
			print('selected',selected)
			print('choice',choice)
			break
		x, y, z = get_next(limit_x, limit_y, limit_z)
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
	radius=0.5 #radius for the chooser
	max_time = 10 #max time to run in seconds, else returns ....?? else shrink radius?
	if num_dim == 3:
		minx=min(data[0])
		maxx=max(data[0])
		miny=min(data[1])
		maxy=max(data[1])
		minz=min(data[2])
		maxz=max(data[2])
		template = read("db/tmp/tmp.json")
		limit_x = {"from": minx, "to":maxx }
		limit_y = {"from": miny, "to":maxy }
		limit_z = {"from": minz, "to": maxz }
		exploration_pt = generate_rand_pt(limit_x, limit_y, limit_z, radius, 1, max_time, data)
		#if fail to find point, shrink radius to half
		while exploration_pt == []:
			exploration_pt = generate_rand_pt(limit_x, limit_y, limit_z, radius*0.5, 1, max_time, data)
		return exploration_pt
	if num_dim == 2:
		minx=min(data[0])
		maxx=max(data[0])
		miny=min(data[1])
		maxy=max(data[1])
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
			exploration_pt = generate_rand_pt(limit_x, limit_y, limit_z, radius*0.5, 1, max_time, data)
			print('im looking for a new point')
			print('radius here',radius)
		print('radius',radius)
		#exploration point is a tuple within a list [(1,2)] so it is converted to a list
	return list(exploration_pt[0])


def meritfunction(data,results,ff_calc):
	#first we need to 'unpack' data and results into arrays
	num_dim=len(data)
	print(num_dim)
	if not (2 <= num_dim <= 3):
		raise ValueError (' utility function for D!=3 OR 2 not completed')
	elif num_dim == 2:
		print('data',data)
		print('results',results)
		#need to pick out above/under & certain frequency for result
		##Check above/under for results. Pick the center frequency

		RBF_Func=Rbf(data[0],data[1],results,function='thin_plate',epsilon=0.001)
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
		#rbf_optima.append(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		#rbf_optima.append(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		rbf_optima.append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		#rbf_optima.append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		#rbf_optima.append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		rbf_optima.append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
	#	rbf_optima.append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		#rbf_optima.append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		#if true, exploit, if false, explore
		current_best_rbf = RBF_max_point_selector(rbf_optima)
		exploit_pt = current_best_rbf['sim point']
		explore_pt = Exploration_point_selector(data,num_dim)
		print('exploit',exploit_pt,'explore',explore_pt[0])
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		X,Y=np.meshgrid(x,y)
		z = RBF_Func(X,Y)
		ax.text(explore_pt[0],explore_pt[1],-1*RBF_TO_OPT(explore_pt),'EXPLORE POINT',fontsize=12)
		ax.text(exploit_pt[0],exploit_pt[1],-1*RBF_TO_OPT(exploit_pt),'EXPLOIT POINT',fontsize=12)
		ax.scatter(exploit_pt[0],exploit_pt[1],-1*RBF_TO_OPT(exploit_pt),s=50,c='r',marker='D')
		ax.scatter(explore_pt[0],explore_pt[1],-1*RBF_TO_OPT(explore_pt),s=50,c='g',marker='D')
		ax.scatter(data[0],data[1],results,alpha=1,c='k')
		ax.plot_surface(X,Y,z,cmap=cm.jet)
		ax.set_xlabel(sys.argv[3])
		ax.set_ylabel(sys.argv[4])
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
		return next_sim, Phase_Selector(data,results,current_best_rbf['Function value']),current_best_rbf

	elif num_dim == 3:
		nfreq=len(results[0][0])
		rbf_optima = {'DE':[[]]*nfreq,
		'DA':[[]]*nfreq,
		'SHGO':[[]]*nfreq
		}
		temp = []
		if ff_calc == "Above":
			for n in range(len(results)):
				temp.append(results[n][0])
		elif ff_calc == "Below":
			for n in range(len(results)):
				temp.append(results[n][1])
		else:
			raise ValueError ('ff_calc incorrect. Do you wish to optimize far field above or below?')
		results_new = temp
		print('results',results)
		print('results_new',results_new)
		#Sort the results from the simulations 
		results_pf=[]
		for n in range(nfreq):
			temp=[]
			print(results_pf)
			for k in range(len(results_new)):
				temp.append(results_new[k][n])
			results_pf.append(temp)
		#Create a list with each element containing a RBF(data,freq). Lowest frequency first in the list.
		RBF_List=[]
		for n in range(nfreq):
			RBF_List.append(Rbf(data[0],data[1],data[2],results_pf[n],function='thin_plate',smooth=0))
		def RBF_min(var):
			x=var[0]
			y=var[1]
			z=var[2]
			return (-1*RBF_Func(x,y,z))
		minx=min(data[0])
		maxx=max(data[0])
		miny=min(data[1])
		maxy=max(data[1])
		minz=min(data[2])
		maxz=max(data[2])
		#print(len(data[0]))
		x = np.linspace(minx,maxx,num=30)
		y = np.linspace(miny,maxy,num=30)
		z = np.linspace(minz,maxz,num=30)
		#local minimizers
		init_guess=np.array([3,3,0.13])
	#	rbf_optima = [[]]*3 #One list for every optimizer. Every sublist will then contain optima for RBF(freq1),RBF(freq2),..,RBF(freqN)
		print(rbf_optima)
		print('len',len(rbf_optima))
		print(RBF_List)
		#def RBF_TO_OPT(x):
		#	return (-1*RBF_Func(x[0],x[1],x[2]))
		for n in range(nfreq):
			RBF_Func = RBF_List[n]
			#The optimizers are minimizers, we then "flip" the RBF Func
			def RBF_TO_OPT(x):
				return (-1*RBF_Func(x[0],x[1],x[2]))
			rbf_optima['DE'][n]=(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
			#rbf_optima['DE'][n].append(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		#	rbf_optima['DE'][n].append(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
			rbf_optima['DA'][n]=(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		#	rbf_optima['DA'][n].append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		#	rbf_optima['DA'][n].append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
			rbf_optima['SHGO'][n]=(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		#	rbf_optima['SHGO'][n].append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		#	rbf_optima['SHGO'][n].append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		#print('evo1',rbf_optima['DE'])
		#print('DA',rbf_optima['DA'])
		#print('SHGO',rbf_optima['SHGO'])
		#Withdraw the best optima from all functions
		rbf_func_max = []
	#	print('DE',rbf_optima['DE'])
		#print('SHGO',rbf_optima['SHGO'])
		print(rbf_optima)
		current_best = {'OPT': 'empty','Function value': -100000, 'sim point':[0,0,0], 'frequency index': 2}
		for opt in rbf_optima:
			for k in range(nfreq):
				print('curr best func',current_best['Function value'])
				if (-1*rbf_optima[opt][k].fun > current_best['Function value']):
					current_best['OPT']=opt
					current_best['Function value']=-1*rbf_optima[opt][k].fun
					current_best['sim point']=rbf_optima[opt][k].x
					current_best['frequency index']=k
		rbf_func_max=current_best['Function value']
		if Phase_Selector(data,results_new,nfreq,rbf_func_max)=='exploit':
			next_sim = current_best['sim point']
			print('next_sim',next_sim)
			print('EXPLOATATIONS','point',next_sim)
		#	next_sim = [next_sim[0][0]]+[next_sim[0][1]]
			plot_color = 'r'
		else:
			next_sim = Exploration_point_selector(data,num_dim)
			print('next_sim',next_sim)
			print('EXPLORATION','point',next_sim)
			next_sim = [next_sim[0][0]]+[next_sim[0][1]]+[next_sim[0][2]]
			plot_color = 'g'
		print(current_best)
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.set_xlabel('pyramid height')
		ax.set_ylabel('pyramid width')
		ax.set_zlabel('source position')
		ax.set_xlim(minx,maxx)
		ax.set_ylim(miny,maxy)
		ax.set_zlim(minz,maxz)
		ax.scatter(current_best['sim point'][0],current_best['sim point'][1],current_best['sim point'][2],color='r')
		plt.show()
		return next_sim






	
	

	
	


	
	
