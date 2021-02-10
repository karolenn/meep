from random import uniform, randint
from .functions import myPoyntingFlux
import matplotlib.pyplot as plt
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
    #print('p1',P_ff1)
    #print('p2',P_ff2)
    for ffpt_i in range(ff_pts):
        tmp.append([a + b for a, b in zip(P_ff1[ffpt_i], P_ff2[ffpt_i])])
    return tmp

def add_poynting_fields_rotated(P_ff1,P_ff2,ff_pts,rotation_integer,nfreq):
    tmp = initialize_poynting_far_field_rotated(ff_pts,nfreq)
    print('here')
    print('rotint',rotation_integer)
    if P_ff1[rotation_integer] == []:
        print('ye')
        print(P_ff1)
        print('rot int',rotation_integer)
    tmp[rotation_integer] = add_poynting_fields(P_ff1[rotation_integer],P_ff2,ff_pts)
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
        rotated_list.append([x_rot,y_rot,z])
    return rotated_list

#initialize poynting field coords
def initialize_poynting_far_field_rotated(ff_pts,nfreq):
    return [[[0]*nfreq]*ff_pts]*6






def unpack_poynting_coords_rotated(initial_coords):
    x = []
    y = []
    z = []
    for n in range(len(list_of_coords)):
        for i in range(len(list_of_coords[0])):
            x.append(list_of_coords[n][i][0])
            y.append(list_of_coords[n][i][1])
            z.append(list_of_coords[n][i][2])
    return x,y,z

def create_far_field_coords(initial_ff_coords):
    x = []
    y = []
    z = []
    for k in range(len(initial_ff_coords)):
        for n in range(6):
            x_ = initial_ff_coords[k][0]
            y_ = initial_ff_coords[k][1]
            z_ = initial_ff_coords[k][2]
            x_rot, y_rot, z_rot = rotate_coordinate(x_,y_,z_,n)
            x.append(x_rot)
            y.append(y_rot)
            z.append(z_rot)
    return x,y,z

#unpack [[x1,y1,z1],....,[xn,yn,zn]] list into [x1,..,xn] and [y1,..,yn] and [z1,..,zn]
def unpack_3_list(list):
    tmp1=[]
    tmp2=[]
    tmp3=[]
    
    for k in range(len(list)):
        tmp1.append(list[k][0])
        tmp2.append(list[k][1])
        tmp3.append(list[k][2])

    return tmp1,tmp2,tmp3

#map x,y,z variables to x,y,z coordinates in meep-space which corresponds to a position on the pyramid wall 
def map_to_pyramid_coords(x,y,z,pyramid_height,simulation_ratio,substrate_ratio):
    inner_pyramid_height = pyramid_height - 0.1
    sh = substrate_ratio*pyramid_height
    sz=pyramid_height*simulation_ratio
        #TODO: Create function that does this mapping
    x_tmp = (inner_pyramid_height*z*math.cos(math.pi/6))/math.tan(62*math.pi/180)
    y_tmp = y*math.tan(math.pi/6)*(inner_pyramid_height*z*math.cos(math.pi/6))/math.tan(62*math.pi/180)
    z_tmp = sz/2-sh-inner_pyramid_height+inner_pyramid_height*z
    return x_tmp,y_tmp,z_tmp


def plot_dipoles_on_pyramids(dipole_positions,rotation_integers,pyramid_height,simulation_ratio,substrate_ratio):

    x_spos,y_spos,z_spos = zip(*dipole_positions)
    ax = plt.axes(projection='3d')
    xspos = []
    yspos = []
    zspos = []
    for n in range(len(x_spos)):
        #TODO: Create function that does this mapping
        x,y,z = map_to_pyramid_coords(x_spos[n],y_spos[n],z_spos[n],pyramid_height,simulation_ratio,substrate_ratio)

        #rotate dipole position
        x,y,z = rotate_coordinate(x,y,z,rotation_integers[n])

        xspos.append(x)
        yspos.append(y)
        zspos.append(z)
    ax.set_xlabel('x-coordinates')
    ax.set_ylabel('y-coordinates')
    ax.scatter(xspos,yspos,zspos)
    plt.title('Dipole positions on pyramid')
    plt.show()

def plot_LEE_convergence(ff_flux,total_flux):

    ratio_1 = []
    ratio_2 = []
    ratio_3 = []
    ratio_4 = []
    ratio_5 = []
    index = []
    for n in range(len(ff_flux)):
        ratio_1.append(100*ff_flux[n][0]/total_flux[n][0])
        ratio_2.append(100*ff_flux[n][1]/total_flux[n][1])
        ratio_3.append(100*ff_flux[n][2]/total_flux[n][2])
        ratio_4.append(100*ff_flux[n][3]/total_flux[n][3])
        ratio_5.append(100*ff_flux[n][4]/total_flux[n][4])
        index.append(n)


    #plt.plot(index,ratio_1,color='red',label='714 nm')
    plt.plot(index,ratio_2,color='orange',label='588 nm')#588 orange
    plt.plot(index,ratio_3,color='g',label='500 nm')#500 g
    plt.plot(index,ratio_4,color='b',label='435 nm')
    #plt.plot(index,ratio_5,color='m',label=' 385 nm')
    plt.legend(loc='best')
    plt.title('LEE convergence')
    plt.grid()
    plt.ylabel('LEE (%)')
    plt.xlabel('Number of dipoles')
    plt.show()

def plot_poynting_coordinates(initial_ff_coords):

    #plot poynting coordinates
    ax2 = plt.axes(projection='3d')
    #unpack initial ff_coords
  
    x_ff,y_ff,z_ff = unpack_3_list(initial_ff_coords)
    ax2.scatter(x_ff,y_ff,z_ff)
    ax2.set_xlabel('x-coordinates')
    ax2.set_ylabel('y-coordinates')
    plt.title('Poynting sampling coordinates')
    plt.show()

def plot_emission_lobe(initial_ff_coords,poynting_total_field,freq):

    x_ff,y_ff,z_ff = unpack_3_list(initial_ff_coords)
    #save all poynting field values (rotated) in the far-field as a list for a given frequency
    ff_values = []
    ff_pts = len(x_ff)
    for n in range(ff_pts):
        val = poynting_total_field[n][freq]
        ff_values.append(val)
    max_val = max(ff_values)
    ff_values_norm = []
    for n in range(ff_pts):
        tmp = ff_values[n]/max_val
        ff_values_norm.append(tmp)
    #plot emission lobe
    ax3 = plt.axes(projection='3d')
 
    X = []
    Y = []
    Z = []
    #the problem is that the x,y,z coords does not match the ff_values, hence the "noise"
    for n in range(len(x_ff)):
        X.append(x_ff[n]*ff_values_norm[n])
        Y.append(y_ff[n]*ff_values_norm[n])
        Z.append(abs(z_ff[n]*ff_values_norm[n]))

    ax3.plot_trisurf(X,Y,Z,cmap='jet',edgecolor='none')
    
    ax3.set_zlabel(r'$ flux')
    ax3.set_title(r'Emission Lobe $\lambda = 500 nm$')
    ax3.set_xlabel('x-coordinates')
    ax3.set_ylabel('y-coordinates')
    plt.show()

def plot_farfield(initial_ff_coords,ff_vals):
    x,y,z=unpack_3_list(initial_ff_coords)
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x,y, ff_vals,cmap='viridis', edgecolor='none')
    ax.set_zlabel(r'$ flux')
    ax.set_title(r'$\lambda = 500 nm$')
    ax.set_xlabel('x-coordinates')
    ax.set_ylabel('y-coordinates')
    plt.show()