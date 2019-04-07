from utils.api import *
from utils.functions import *
from lib.pyramid import Pyramid
from src.optimizer import Optimizer

def main(sim_name, times):
    
    for _ in range(times):
        db = read("db/results/{}.json".format(sim_name))
        if db == None:
            print("reading from initial_results")
            db = read("db/initial_results/{}.json".format(sim_name))
            if db == None:
                print("could not open db/initial_results/{}.json".format(sim_name))
                exit(0)
            else: 
                write_result("db/results/{}.json".format(sim_name), db)

        results = db_to_array(db)
        print(results)
        # config = utility_function(results)

        # pyramid = Pyramid(config["pyramid"])
        # result = pyramid.simlate(config["simulate"])

        data = sim_to_json(db, ["kalle",  False, True])
        write_result("db/results/{}.json".format(sim_name), data)







if __name__ == "__main__":
    import sys
    if (len(sys.argv)) == 2:
        main(sys.argv[1], 10)
    elif(len(sys.argv)) == 3:
        main(sys.argv[1], sys.args[2])
    else:
        print("Not enough arguments")
        exit(0)
