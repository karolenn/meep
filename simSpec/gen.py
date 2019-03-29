import os
import numpy as np
file_name = 'pyramid'

def frange(start, stop, step):
	i = start
	while i < stop:
		yield i
		i +=step
for i in np.linspace(2,3,11):
	for j in np.linspace(2,3,11):
		os.system('echo r: {} time: {} s_pos: {} p_height: {} p_width: {} pml: {} >> {}.in'.format( 40, 15, 0.08, i, j, 0.1, file_name))

