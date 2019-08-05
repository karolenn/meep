from utils.api import *
#from lib.pyramid import Pyramid
#from src.optimizer import Optimizer
from initial_runs import initial_runs
from utility_function import merit_function
from src.execute_simulation import execute_simulation

def main(sim_name,number_of_runs,ff_calc,args):

    #Perform the initial runs and write the results to the database
   # initial_runs(sim_name)
    db = read("db/sim_spec/{}.json".format(sim_name))
    for config in db:
        execute_simulation(config, sim_name)
    #"place below in a for loop"
    #Take the initial run results, calculate the next input parameters for the simulation
    for n in range(int(number_of_runs)):
        next_run = merit_function(sim_name,ff_calc,args)
        #execute_sim needs db entry for next run
        execute_simulation(next_run,sim_name)
    
#What is missing is that we need to save for each iteration if we performed exploit or explore run. Optionally how the optimizers
#worked for the exploit result for each iteration in main. 




if __name__ == "__main__":
    import sys
    if (len(sys.argv)) > 3:
       main(sys.argv[1], sys.argv[2], sys.argv[3],
        [arg.strip() for arg in sys.argv[4:]])
    else:
        print("Not enough arguments")
        exit(0)
