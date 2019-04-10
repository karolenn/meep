from utils.api import *
from lib.pyramid import Pyramid

def initial_runs(sim_name):

    db = read("db/sim_spec/{}.json".format(sim_name))
    if db == None:
        print("could not open db/sim_spec/{}.json".format(sim_name))
        exit(0)
    
    pyramid = Pyramid({'source_position': 0.06, 'pyramid_height': 2.2, 'pyramid_width': 2, 'source_direction': 'mp.Ey', 'frequency_center': 2, 'frequency_width': 0.5, 'number_of_freqs': 1, 'cutoff': 2}
)
    for config in db:
        pyramid = Pyramid(config["pyramid"])
        print(config["pyramid"])
        result = pyramid.simulate(config["simulate"])
        data = sim_to_json(config, result)
        write_result("db/initial_results/{}.json".format(sim_name), data)
    


if __name__ == "__main__":
    import sys
    if (len(sys.argv)) == 2:
        initial_runs(sys.argv[1])
    else:
        print("Not enough arguments")
        exit(0)