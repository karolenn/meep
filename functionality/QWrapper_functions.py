from random import uniform, randint
from .functions import myPoyntingFlux
from collections import deque
import math as math




###Functionality for the Qwrapper in order to simulate quantum well emission

#this function linearly adds E,H fields with weights wx, wy, wz for a 3-group which corresponds due to linearity to the pyramids polarization, so (wx=1,wy=0,wz=0) is x polarized dipole
#TODO:We might need to convert this to a faster method with increasing ffpts
def linear_combine_fields(f1,f2,f3,wx,wy,wz,f_pts):
    tmp = []
    #f is a list of lists of length ff_pts
    for ffpt_i in range(f_pts):
        tmp.append([wx*px + wy*py + wz*pz for px, py, pz in zip(f1[ffpt_i], f2[ffpt_i], f3[ffpt_i])])
    return tmp

def add_poynting_fields(P_ff1,P_ff2,ff_pts):
    tmp = []
    for ffpt_i in range(ff_pts):
        tmp.append([a + b for a, b in zip(P_ff1[ffpt_i], P_ff2[ffpt_i])])
    return tmp



#Calculate the poynting scalar field for a 3-group 
#works for 1 pyramid
def calculate_poynting_field(summed_ff,ff_pts,nfreq):
    tmp = []
    ff_at_pt = []
    for ffpt_i in range(ff_pts):
        ff_at_pt = summed_ff[ffpt_i]
   #     print('ff_at_pt',ff_at_pt)
        poynting_vector_at_pt=[]
        i=0
        for freq in range(nfreq):
            Pr = myPoyntingFlux(ff_at_pt,i)
            poynting_vector_at_pt.append(Pr)
        #    print('vec',poynting_vector_at_pt)
            i = i + 6 #to keep track of the correct index in the far-field array for frequencies, (freq=1) Ex1,Ey1,..,Hx1,..Hz1,Ex2,...,Hz2,..
        tmp.append(poynting_vector_at_pt)
    return tmp

def rotate_field(field,rotation,ff_pts):
    #n = 0 for no rotation, n=1 60 degrees,.., n=5 300 degrees.
    n = randint(0,5)

def rotate_coordinate(x,y,z,rotation_integer):
    n = rotation_integer
  #  print('--')
  #  print(x,y)
    rotation_pos_x = x*math.cos(n*math.pi/3)-math.sin(n*math.pi/3)*y
    rotation_pos_y = x*math.sin(n*math.pi/3)+math.cos(n*math.pi/3)*y
  #  print(rotation_pos_x,rotation_pos_y)
   # print('--')

    return rotation_pos_x,rotation_pos_y,z

#rotate x,y position for a list of coords [[x1,y1,z1],[x2,y2,z2],...,[xi,yi,zi]]
def rotate_coordinate_list(list_of_coords,rotation_integer):

    len_of_list = len(list_of_coords)

    n = rotation_integer

    rotated_list = []
    for i in range(len_of_list):
        x = list_of_coords[i][0]
        y = list_of_coords[i][1]
        z = list_of_coords[i][2]
        x_rot = x*math.cos(n*math.pi/3)-math.sin(n*math.pi/3)*y
        y_rot = x*math.sin(n*math.pi/3)+math.cos(n*math.pi/3)*y
        z = z
        rotated_list.append([x,y,z])

    return rotated_list


