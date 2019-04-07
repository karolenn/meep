from utils.api import *
#from lib.pyramid import Pyramid
from src.optimizer import Optimizer

def main(sim_name,sim_id):
    
    db = read("db/result/{}.{}.json".format(sim_name, sim_id))
    if db == None:
        print("could not open db/result/{}.{}.json".format(sim_name,sim_id))
        exit(0)
        
    util_imput = db_to()

    while True:
        sim_spec = util(util_imput)
        pyramid = Pyramid(sim_spec)
        result = pyramid.simlate()

        sim_id += 1
        util_imput = Optimizer(result)
        write("db/result/{}.{}.json".format(sim_name, sim_id), data)
        data = sim_to_json(result,util_imput)

            
        #write_utils("db/utils/{}.{}.json".format(sim_name,sim_id), data)

        #data = opt_to_json(opt_results)


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
