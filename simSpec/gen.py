import os
file_name = 'pyramid'
for i in range(1,20, 1):
	os.system('echo res: {} simTime: {} >> {}.in'.format( i, 1, file_name))

