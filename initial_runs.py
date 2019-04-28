from utils.api import *
from src.execute_simulation import execute_simulation

def initial_runs(sim_name):

    db = read("db/sim_spec/{}.json".format(sim_name))
    if db == None:
        print("could not open db/sim_spec/{}.json".format(sim_name))
        exit(0)

    for config in db:
        execute_simulation(config, sim_name)


if __name__ == "__main__":
    import sys
    if (len(sys.argv)) == 2:
        initial_runs(sys.argv[1])
    else:
        print("Not enough arguments")
        exit(0)
