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

def Unpacker(data):
	num_dim=len(data)
	if num_dim == 0:
		print('Error: 0 dims to RBF')
	elif num_dim == 1:
		return(Rbf(data[0],sum(results,[])))
	elif num_dim == 2:
		return(Rbf(data[0],data[1],sum(results,[])))
	elif num_dim == 3:
		return(Rbf(data[0],data[1],data[2],sum(results,[])))
	elif num_dim == 4:
		return(Rbf(data[0],data[1],data[2],data[3],sum(results,[])))
	else:
		print('Error: merit function to many inparameters')

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

#används ej?
def mindistfunction(args):
	bounds = [min(args),max(args)]
	max_dist_pt=0

	return max_dist_pt

#används ej?
def distfunction(x,args):
	num_variables = len(args)
	#if x outside bounds
	#return 0
	func_value = []
	for n in range(num_variables):
		func_value.append(abs(x-args[n]))
	return min(func_value)


def meritfuncshort(data,results):
	num_dim = len(data)
	if num_dim == 0:
		print('Error: 0 dims to RBF')
	elif num_dim == 1:
		print('1')
	elif num_dim == 2:
		print('2')
	elif num_dim == 3:
		RBF_Func=Rbf(data[0],data[1],data[2],sum(results,[]),function='thin_plate',smooth=0)
	else:
		print('too many args in merit function')

#Choose if we're going for exploration or explotation phase
def Phase_Selector(data,results,rbf_optima):
	if Compare_max_res_and_rbf(data,results,rbf_optima) == True:
		return ('exploit')
	else:
		return ('explore')

#compares difference between rbf_maximum and simulated maximum, returns True if we're exploting. False if we're exploring.
def Compare_max_res_and_rbf(data,results,rbf_optima):
	#plocka ut högsta simulerade målfunktionsvärdet
	maxf=max(results)
	print('max func value',maxf)
	#plocka ut max-funktionsvärdet från varje lösare
	rbf_func_max = []
	for n in range(len(rbf_optima)):
		rbf_func_max.append(-1*rbf_optima[n].fun)
	maxrbf = max(rbf_func_max)
	print('max rbf value',maxrbf)
	print('difference:',maxrbf-maxf)
	if maxrbf-maxf > 0.05:
		return True
	else:
		return False

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
		limit_x = {"from": minx, "to":maxx }
		limit_y = {"from": miny, "to":maxy }
		limit_z = {"from": minz, "to": maxz }
		exploration_pt = generate_rand_pt(limit_x, limit_y, limit_z, radius, 1, max_time, data)
		#if fail to find point, shrink radius to half
		while exploration_pt == []:
			exploration_pt = generate_rand_pt(limit_x, limit_y, limit_z, radius*0.5, 1, max_time, data)
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
	return exploration_pt

def RBF_max_point_selector(rbf_optima):
	rbf_func_max = []
	#withdraw all function value optimas from the rbf
	for n in range(len(rbf_optima)):
		rbf_func_max.append(-1*rbf_optima[n].fun)
	#Select the largest function value from the rbf and save its index
	rbf_func_max_index = max(range(len(rbf_func_max)), key=rbf_func_max.__getitem__)
	rbf_func_pt = []
	#withdraw all the function variables from the rbf for the corresponding f value optima
	for n in range(len(rbf_optima)):
		print(list(rbf_optima[n].x))
		rbf_func_pt.append(list(rbf_optima[n].x))
	#return the pt/variables corresponding to the highest rbf func value optima
	print('rbf func max',rbf_func_max)
	print('rbf funct pt',rbf_func_pt)
	print('index',rbf_func_max_index)
	testguy=rbf_func_pt[1]
	print('hereeee',rbf_func_pt[0],type(testguy))
	return rbf_func_pt[rbf_func_max_index]


def meritfunction(data,results):
	#first we need to 'unpack' data and results into arrays
	num_dim=len(data)
	if num_dim == 0:
		print('Error: 0 dims to RBF')
	elif num_dim == 1:
		#remove duplicates
		data = np.unique(data).tolist()
		results = sum(results,[])
		results = np.unique(results).tolist()
		print('x:',data,len(data))
		print('f:',results,len(results))
		RBF_Func=Rbf(data,results)
		minx=min(data)
		maxx=max(data)
		minf=min(results)
		maxf=max(results)
		x = np.linspace(minx,maxx,num=30)
		f = np.linspace(minf,maxf,num=30)
		#optimizer
		def RBF_min(var):
			x = var[0]
			return (-1*RBF_func(x))
		init_guess = np.array([(maxx-minx)/2])
		bounds_RBF = Bounds([minx, maxx])
		rbf_max = minimize(RBF_min,x0=init_guess, method='COBYLA', options={'verbose': 1}, bounds=bounds_RBF)

		#2d plot
		fig = plt.figure()
		plt.plot(data[0],results)
		plt.plot(x,RBF_Func(x))
		plt.show()
	elif num_dim == 2:
		print(data)
		print(results)
		RBF_Func=Rbf(data[0],data[1],sum(results,[]),function='thin_plate')
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
		rbf_optima.append(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		rbf_optima.append(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		rbf_optima.append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		rbf_optima.append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		rbf_optima.append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		rbf_optima.append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		rbf_optima.append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		rbf_optima.append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy]]))
		#if true, exploit, if false, explore
		if Phase_Selector(data,results,rbf_optima)=='exploit':
			next_sim = RBF_max_point_selector(rbf_optima)
			print('next_sim',next_sim)
			print('EXPLOATATIONS','point',next_sim)
		#	next_sim = [next_sim[0][0]]+[next_sim[0][1]]
			plot_color = 'r'
		else:
			next_sim = Exploration_point_selector(data,num_dim)
			print('next_sim',next_sim)
			print('EXPLORATION','point',next_sim)
			next_sim = [next_sim[0][0]]+[next_sim[0][1]]
			plot_color = 'g'
		#3d plot
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		X,Y=np.meshgrid(x,y)
		z = RBF_Func(X,Y)
		print('hehe',next_sim)
		ax.text(next_sim[0],next_sim[1],-1*RBF_TO_OPT(next_sim),'POINT',fontsize=12)
		ax.scatter(next_sim[0],next_sim[1],-1*RBF_TO_OPT(next_sim),s=50,c=plot_color,marker='D')
		ax.scatter(data[0],data[1],sum(results,[]),alpha=1,c='k')
		ax.plot_surface(X,Y,z,cmap=cm.jet)
		ax.set_xlabel(sys.argv[2])
		ax.set_ylabel(sys.argv[3])
		plt.show()
		return next_sim

	elif num_dim == 3:
		print(data[0])
		print(data[1])
		print(data[2])
	#	print(sum(results,[]))
		data.append(sum(results,[]))
		bar = defaultdict(list)
		for x,y,z,r in zip(*data):
			bar[z].append((x,y,r))
		bar = sorted(bar.items())

		RBF_Func=Rbf(data[0],data[1],data[2],sum(results,[]),function='thin_plate',smooth=0)
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
		rbf_optima = []
		rbf_optima.append(minimize(RBF_min,x0=init_guess, method='L-BFGS-B',bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(minimize(RBF_min,x0=init_guess, method='TNC',bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(minimize(RBF_min,x0=init_guess, method='SLSQP',bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		#print('bfgs',rbf_optima[0])
		#print('tnc',rbf_optima[1])
		#print('slsqp',rbf_optima[2])
		#global minimizers
		def RBF_TO_OPT(x):
			return (-1*RBF_Func(x[0],x[1],x[2]))
		rbf_optima.append(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(differential_evolution(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(shgo(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		rbf_optima.append(dual_annealing(RBF_TO_OPT,bounds=[[minx,maxx],[miny,maxy],[minz,maxz]]))
		print('evo1',rbf_optima[3])
		print('evo2',rbf_optima[4])
		print('evo3',rbf_optima[5])
		print('shgo1',rbf_optima[6])
		print('shgo2',rbf_optima[7])
		print('shgo3',rbf_optima[8])
		print('dual anneal1',rbf_optima[9])
		print('dual anneal2',rbf_optima[10])
		print('dual anneal3',rbf_optima[11])


		X,Y,Z=np.meshgrid(x,y,z)
	#	print('X',X)
		f = RBF_Func(X,Y,Z)

		X,Y=np.meshgrid(x,y)

		n=0
		for _source , _slice in bar:
		#	print('source',_source)
			#print()
		#	print(f[n][:][:])
			x,y,r = zip(*_slice)
			ax = plt.axes(projection='3d')
			ax.plot_surface(X,Y,f[:,:,n],cmap=cm.jet)
			n+=1
			#print('x',x)
			#print('y',y)
			#print('r',r)
			ax.scatter(x,y,r)
			plt.show()
	elif num_dim == 4:
		RBF_Func=Rbf(data[0],data[1],data[2],data[3],sum(results,[]))
	else:
		print('Error: merit function to many inparameters')
#	approx_func=Rbf((Unpacker(data,n) for n in range(num_dim)),(Unpacker(results,m) for m in range(num_dim)))
	#approx_func=Rbf((data[n] for n in range(len(data))), (results[m] for m in range(len(results))))

	# x=data[0]
	# y=data[1]
	# approx_func=Rbf(data)
	# x,y=np.linspace(2,3)
	# #3dplot
	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# #ax.scatter(x,y,z,cmap=cm.jet)
	# ax.plot_surface(x,y,approx_func)
	# plt.show()







	
	

	
	


	
	
