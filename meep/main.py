from utils.api import *
#from lib.pyramid import Pyramid
from src.optimizer import Optimizer

def main(sim_name,sim_id):
    
    db = read("db/sim_spec/{}.{}.json".format(sim_name, sim_id))
    if db == None:
        print("could not open db/sim_spec/{}.{}.json".format(sim_name,sim_id))
        exit(0)
    
    sim_results = []
    for sim_spec in db:
        imp = json_to_imp(sim_spec)
        pyramid = Pyramid(imp)
        result = pyramid.simlate()
        opt = Optimizer(result)
        data = sim_to_json(result, opt)
        write_result("db/result/{}.{}.json".format(sim_name, sim_id), data)


    

    opt_results = []
    for result in sim_results:
    
    new_imput = util()
    write_utils("db/utils/{}.{}.json".format(sim_name,sim_id), data)

    sim_id += 1
    data = opt_to_json(opt_results)


class Pyramid():
    def __init__(self, imp):
        self.imp = imp

    def simlate(self):
        return self.imp






if __name__ == "__main__":
    import sys
    if (len(sys.argv)) == 2:
        main(sys.argv[1], 0)
    elif (len(sys.argv)) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        print("Not enough arguments")
        exit(0)
