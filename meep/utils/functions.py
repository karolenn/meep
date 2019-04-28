import math as math
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import Bounds
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys


###FUNCTIONS##########################################################
def fibspherepts(r,theta,npts,xPts,yPts,zPts):

	offset=0.75/npts #this is hardcoded for angle = pi/6
	range_npts=int((theta/math.pi)*npts)
	increment = math.pi*(3 - math.sqrt(5))

	for n in range(range_npts):
		zPts.append(r*((n*offset-1)+(offset/2)))
		R = r*math.sqrt(1-pow(zPts[n]/r,2))
		phi = (n % npts)*increment
		xPts.append(R*math.cos(phi))
		yPts.append(R*math.sin(phi))

	return(xPts,yPts,zPts)
	

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

def mindistfunction(args):
	bounds = [min(args),max(args)]
	max_dist_pt=0

	return max_dist_pt

def distfunction(x,args):
	num_variables = len(args)
	#if x outside bounds
	#return 0
	func_value = []
	for n in range(num_variables):
		func_value.append(abs(x-args[n]))
	return min(func_value)




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
		RBF_func=Rbf(data[0],data[1],sum(results,[]),function='thin_plate')
		minx=min(data[0])
		maxx=max(data[0])
		miny=min(data[1])
		maxy=max(data[1])
		minf=min(results)
		maxf=max(results)
		index_max = np.argmax(results)
		print(minx,maxx,miny,maxy,minf,maxf)
		x = np.linspace(minx,maxx,num=30)
		y = np.linspace(miny,maxy,num=30)
		f = np.linspace(minf,maxf,num=30)
		#optimizer
		def RBF_min(var):
			x=var[0]
			y=var[1]
			return (-1*RBF_func(x,y))
		init_guess = np.array([(maxx-minx)/2,(maxy-miny)/2])
		init_guess=np.array([data[0][index_max],data[1][index_max]])
		rbf_max = minimize(RBF_min,x0=init_guess, method='L-BFGS-B', options={'verbose': 1},bounds=[[minx,maxx],[miny,maxy]])
		#3d plot
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		X,Y=np.meshgrid(x,y)
		z = RBF_func(X,Y)
		next_sim = rbf_max.x
	#	print(X)
	#	print(Y)
		ax.scatter(rbf_max.x[0],rbf_max.x[1],-1*rbf_max.fun,s=50,c='r',marker='D')
		ax.scatter(data[0],data[1],sum(results,[]),alpha=1,c='k')
		ax.plot_surface(X,Y,z,cmap=cm.jet)
		ax.set_xlabel(sys.argv[2])
		ax.set_ylabel(sys.argv[3])
		plt.show()
		return rbf_max.x


	elif num_dim == 3:
		RBF_Func=Rbf(data[0],data[1],data[2],sum(results,[]))
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







	
	

	
	


	
	
