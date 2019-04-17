import math as math


###FUNCTIONS##########################################################
def fibspherepts(r,theta,npts,xPts,yPts,zPts):

	offset=2/npts
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



	
	

	
	


	
	
