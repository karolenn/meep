from functionality.api import *
from initial_runs import initial_runs
from functionality.utility_function import merit_function
from execute.execute_simulation import execute_simulation

#Performs initial runs and then performs 'number_of_runs' additional simulation evaluations with the each simulation decided by the merit function
#Usage type example: python main db N S "Above" pyramid_width source_position to run S types using N cores simulating far field above the pyramid and varying pyramid width and source position

def main(sim_name,cores,number_of_runs,ff_calc,args):
    #Perform the initial runs and write the results to the database
    initial_runs(sim_name)
    #Take the initial run results, calculate the next input parameters for the simulation
    for n in range(int(number_of_runs)):
        next_run = merit_function(sim_name,ff_calc,args)
        #execute_sim needs db entry for next run
        execute_simulation(next_run,sim_name,cores)
    


if __name__ == "__main__":
    import sys
    if (len(sys.argv)) > 3:
       main(sys.argv[1], sys.argv[2], sys.argv[3],
        [arg.strip() for arg in sys.argv[4:]])
    else:
        print("Not enough arguments")
        exit(0)
