import subprocess
from functionality.api import write


#Nessecary to keep python from abusing RAM when running several simulations in row
#Tried using python garbage collectors, RAM clearing functions, explicitly removing pyramid class objects after each run, but RAM did not properly clear after each simulation
#Hence, a work around was found using subprocess
def execute_simulation(config, output_file, cores):
    write("db/tmp/tmp.json", config)
    print("#######################EXECUTE SIMULATION STARTED#######################")
    subprocess.run(['mpirun', "-np", "{}".format(cores), "python3", "structure/pyramid.py", output_file])
    print("#######################EXECUTE SIMULATION ENDED#########################")
    print("")
