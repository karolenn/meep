from utils.api import *
#from lib.pyramid import Pyramid
from src.optimizer import Optimizer
from initial_runs import initial_runs
from utility_function import merit_function
from src.execute_simulation import execute_simulation

def main(sim_name,number_of_runs,args):

    #Perform the initial runs and write the results to the database
    initial_runs(sim_name)
    #"place below in a for loop"
    #Take the initial run results, calculate the next input parameters for the simulation
    for n in range(int(number_of_runs)):
        next_run = merit_function(sim_name,args)
        #execute_sim needs db entry for next run
        execute_simulation(next_run,sim_name)
    
    "run next run"


""" def main(sim_name):
    
    db = read("db/sim_spec/{}.json".format(sim_name))
    if db == None:
        print("could not open db/sim_spec/{}.json".format(sim_name))
        exit(0)
    
    sim_results = []
    for sim_spec in db:
        imp = json_to_imp(sim_spec)
        pyramid = Pyramid(imp)
        result = pyramid.simlate()
        opt = Optimizer(result)
        data = sim_to_json(result, opt)
        write_result("db/result/{}.json".format(sim_name), data)


    

    opt_results = []
    for result in sim_results:
    
    new_imput = util()
    write_utils("db/utils/{}.json".format(sim_name), data)

    sim_id += 1
    data = opt_to_json(opt_results)


class Pyramid():
    def __init__(self, imp):
        self.imp = imp

    def simlate(self):
        return self.imp
 """




if __name__ == "__main__":
    import sys
    if (len(sys.argv)) > 3:
       main(sys.argv[1], sys.argv[2],
        [arg.strip() for arg in sys.argv[3:]])
    else:
        print("Not enough arguments")
        exit(0)
