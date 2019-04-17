from utils.api import *
from utils.functions import *
from lib.pyramid import Pyramid
from src.optimizer import Optimizer
import numpy as np

def merit_function(sim_name, **args):
    db = read("db/initial_results/{}.json".format(sim_name))
    "withdraw the (**args) source_pos, pyramid size etc from initial results and store them in array"
    values = []
    for n in args: #n is index in db
        print(n)
        print(db_to_array(db,"pyramid",n))
        values.append(db_to_array(db,"pyramid",n))
    "withdraw the flux_ratio results from initial_runs simulations"
    sim_results=db_to_array(db,"result","flux_ratio")
    print(sim_results)
    print(values) #What remains to do is to send this badboy to an actual merit function that computes the next sim
    meritfunction(values,sim_results)
    


    # for _ in range(times):
    #     db = read("db/results/{}.json".format(sim_name))
    #     if db == None:
    #         print("reading from initial_results")
    #         db = read("db/initial_results/{}.json".format(sim_name))
    #         if db == None:
    #             print("could not open db/initial_results/{}.json".format(sim_name))
    #             exit(0)
    #         else: 
    #             write("db/results/{}.json".format(sim_name), db)

    #     results = db_to_array(db)
    #     print(results)
    #     # config = utility_function(results)

    #     # pyramid = Pyramid(config["pyramid"])
    #     # result = pyramid.simlate(config["simulate"])

    #     data = sim_to_json(db[0], ["kalle",  "False", "True"])
    #     write_result("db/results/{}.json".format(sim_name), data)





"usage python utility_function.py 02 source_position=0 pyramid_height=1 pyramid_width=2"


if __name__ == "__main__":
    import sys
    if(len(sys.argv)) < 12:
        merit_function(sys.argv[1],
        **dict(arg.split('=') for arg in sys.argv[2:]))
    else:
        print("Not enough arguments")
        exit(0)
