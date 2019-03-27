#!/usr/bin/env python3
import sys
import os
import re
import time

RED = '\x1b[38;5;1m'
GREEN = '\x1b[38;5;2m'
NULL = '\x1b[0m'
BLUE = '\x1b[1;36m' 

file_name = "pyramid"

simSpec_dir = 'simSpec'
start = time.time() 
testing_items = os.listdir(simSpec_dir)
def ends_in(x):
    return lambda string: string.endswith(x)
ins = list(sorted(filter(ends_in('.in'), testing_items)))
try:
    os.mkdir('results')
except OSError:
    pass

def test_for(in_file):
    with open(os.path.join(simSpec_dir,in_file), 'r') as f:
        count = 0
        for line in f:
            print("\r simulation: {}".format(count) , end="")
            count +=1
            try:
                g = re.match(r"res:\s*(\-*\d+)\s*sim_time:\s*(\-*\d+)\s*source_pos:\s*(\-*\d+\.\d+)\s*p_height:\s*(\-*\d+\.*\d*)\s*p_width:\s*(\-*\d+\.*\d*)\s*dpml:\s*(\-*\d+\.*\d*)", line, re.M|re.I )
                args1 = g.group(1)
                args2 = g.group(2) 
                args3 = g.group(3)
                args4 = g.group(4)
                args5 = g.group(5)
		args6 = g.group(6)
                os.system('mpirun -np 3 python {}.py {} {} {} {} {} {} |grep "Total_Flux:*\|Elapsed run time*" >> results/{}.out'.format(file_name, args1, args2, args3, args4, args5, args6, in_file.replace(".in",'')))
            except :
                print("\n{}Failed {} - {} In file: {} {} {} At row: {}{}".format(RED, NULL, BLUE, NULL, in_file, BLUE, NULL,  count))

for in_file in ins:
    test_for(in_file)
    print("\n{}{} - Done {}".format(in_file,GREEN, NULL))

print("\n{}✔ - Simulations Done{} in {:.2f}s".format(GREEN, NULL, time.time()- start))

