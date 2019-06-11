import subprocess
from utils.api import *

def execute_simulation(config, output_file):
    write("db/tmp/tmp.json", config)
    subprocess.run(['mpirun', "-np", "2", "python3", "lib/pyramid.py", output_file])
    


