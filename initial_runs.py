from functionality.api import read
from execute.execute_simulation import execute_simulation

###This script reads a sim specification file and simulates each structure in the given db 
###To run, type python initial_runs db n where db is the sim_spec file to run and n is the number of cores that MPIRUN use


def initial_runs(sim_name,cores):

    db = read("db/sim_spec/{}.json".format(sim_name))
    if db == None:
        print("could not open db/sim_spec/{}.json".format(sim_name))
        exit(0)

    for config in db:
        execute_simulation(config, sim_name, cores)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 :
        print(sys.argv)
        print([arg.strip() for arg in sys.argv[1:]])
        initial_runs(sys.argv[1],sys.argv[2])
    else:
        print("Specify database to run and the number of CPU cores to use")
        exit(0)
