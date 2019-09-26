from functionality.api import *
from functions import utility_function


###This function only processes data so it can be passed and retrieved from the utility function that decides the next point to simulate 
###in the optimization schema
#usage python utility_function.py 02 "Above/Below" source_position pyramid_height pyramid_width

def pass_data_utility_function(sim_name, ff_calc, args):
    db = read("../db/initial_results/{}.json".format(sim_name))
    #withdraw the (**args) source_pos, pyramid size etc from initial results and store them in array"
    values = []
    for n in args: #n is index in db
        values.append(db_to_array(db,"pyramid",n))
    #withdraw the flux_ratio results from initial_runs simulations"
    sim_results=db_to_array(db,"result","flux_ratio")
    #Withdraw ff_ratio above or under from results. 
    #values and sim results are now arrays in arrays. Inner array is data from single simulation
    #We enter the if statement if we only calculated flux above or under
    if len(sim_results[0]) != 1:
        sim_results=process_results(sim_results,ff_calc)

    #Pass simulation results to utility function and returns next simulation to run along with other data
    result, max_result, util_data, rbf_opt = utility_function(values,sim_results, ff_calc)
    #write optimizer and explore/exploit info to database
    write_result("db/results/{}.json".format(sim_name), [result,"max result:",max_result,util_data,rbf_opt])
    template = read("db/tmp/tmp.json")
    for i, name in enumerate(args):
        print(i,result[i])
        template["pyramid"][name] = result[i]
    if "pyramid_height" not in args:
        template["pyramid"]["pyramid_height"]=template["pyramid"]["pyramid_width"]*tan(pi*62/180)/2
    return template








if __name__ == "__main__":
    import sys
    if 5 <= (len(sys.argv)) <= 7:
        print(sys.argv)
        pass_data_utility_function(sys.argv[1], sys.argv[2],
        [arg.strip() for arg in sys.argv[3:]])
    else:
        print("Incorrect arguments passed in pass_data_utility_function in folder functionality/utility_function.py")
        exit(0)
