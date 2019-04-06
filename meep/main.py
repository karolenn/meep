from src.pyramid import Pyramid
import sys

if (len(sys.argv)) != 7:
		print("Not enough arguments")
		exit(0)

resolution= int(sys.argv[1])					#resolution of the pyramid. Measured as number of pixels / unit distance
simulation_time=int(sys.argv[2])				#simulation time for the sim. #Multiply by a and divide by c to get time in fs.
source_pos=float(sys.argv[3])					#pos of source measured	measured as fraction of tot. pyramid height from top. 
pyramid_height=float(sys.argv[4])				#height of the pyramid in meep units 3.2
pyramid_width=float(sys.argv[5])					#width measured from edge to edge 2.6
dpml=float(sys.argv[6])
pyramid = Pyramid(debug = True)
pyramid.simulate(resolution, simulation_time, source_pos, pyramid_height, pyramid_width, dpml)