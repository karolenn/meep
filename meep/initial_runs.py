from utils.api import *
from lib.pyramid import Pyramid

def initial_runs(sim_name):

    db = read("db/sim_spec/{}.json".format(sim_name))
    if db == None:
        print("could not open db/sim_spec/{}.json".format(sim_name))
        exit(0)
    
    for config in db:
        pyramid = Pyramid(config["pyramid"])
        result = pyramid.simulate(config["simulate"])
        data = sim_to_json(config, result)
        write_result("db/initial_result/{}.json".format(sim_name), data)
    


if __name__ == "__main__":
    import sys
    if (len(sys.argv)) == 2:
        initial_runs(sys.argv[1])
    else:
        print("Not enough arguments")
        exit(0)