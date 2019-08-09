from utils.api import *
from utils.functions import *
from lib.pyramid import Pyramid
from math import tan
from math import pi
#from src.optimizer import Optimizer
import numpy as np

#We want to pass RBF_Func and rbf_optima to main so we can compare the RBF and optimas over many iterations
def merit_function(sim_name, ff_calc, args):
    db = read("db/initial_results/{}.json".format(sim_name))
    "withdraw the (**args) source_pos, pyramid size etc from initial results and store them in array"
    values = []
    for n in args: #n is index in db
        values.append(db_to_array(db,"pyramid",n))
    "withdraw the flux_ratio results from initial_runs simulations"
    sim_results=db_to_array(db,"result","flux_ratio")
    #Withdraw ff_ratio above or under from results. 
 #   print('sim results passed to process',sim_results)
    if len(sim_results[0]) != 1:
        sim_results=process_results(sim_results,ff_calc)
    #values and sim results are now arrays in arrays. Inner array is data from single simulation
  #  meritfunction(values,sim_results)
    result, max_result, util_data, rbf_opt = meritfunction(values,sim_results, ff_calc)
    #write optimizer and explore/exploit info to database
    write_result("db/results/{}.json".format(sys.argv[1]), [result,"max result:",max_result,util_data,rbf_opt])
    template = read("db/tmp/tmp.json")
  #  print('args',args)
   # print('result from merit func',result)
    for i, name in enumerate(args):
        template["pyramid"][name] = result[i]
    template["pyramid"]["pyramid_height"]=template["pyramid"]["pyramid_width"]*tan(pi*62/180)/2
    return template





#usage python utility_function.py 02 "Above/Below" source_position pyramid_height pyramid_width


if __name__ == "__main__":
    import sys
    if(len(sys.argv)) < 12:
        merit_function(sys.argv[1], sys.argv[2],
        [arg.strip() for arg in sys.argv[3:]])
    else:
        print("Not enough arguments")
        exit(0)
