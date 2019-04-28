import subprocess
from utils.api import *

def exicute_simulation(config, output_file):
    write("db/tmp/tmp.json", config)
    subprocess.run(['mpirun', "-np", "4", "python3", "lib/pyramid.py", output_file])
    


