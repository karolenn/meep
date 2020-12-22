from functionality.api import read, db_to_array, polar_to_complex_conv, complex_to_polar_conv
from functionality.functions import myPoyntingFlux,PolarizeInPlane, draw_uniform,rotate_list,rotate_list2
from functionality.QWrapper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import math as math
from random import uniform, randint
    #keeps track on number of emission positions and dipole arrangement per position


def Qwell_wrapper(sim_name,number_of_dipoles):

    #load simulation data
    db = read("db/initial_results/{}.json".format(sim_name))
    if db == None:
        print("could not open db/initial_results/{}.json".format(sim_name))
        exit(0)
    #load simulation parameters
    number_of_pyramids=len(db)

    #Natural assumption is that sampling is according to the modified spherical sampling algorithm
    #add +1 for the (0,0,r) point
    ff_pts = db[0]["simulate"]["ff_pts"]
    npts = ff_pts
    range_npts = int(ff_pts*ff_pts*2+1)
    theta = math.pi/db[0]["simulate"]["ff_angle"]
    fibb_sampling = False
    if fibb_sampling:
            if theta == math.pi/6:
                npts = ff_pts*3
            else:
                npts = ff_pts
            range_npts=int((theta/math.pi)*npts) 


    ff_pts = range_npts

    nfreq=db[0]["pyramid"]["number_of_freqs"]

    #array to save resulting E,H far-fields in
    all_ffields = []

    #Withdraw all the far fields from all the pyramids in active db
    #in db fields are stored in polar it is converted to complex here as well
    total_time = 0
    for pyramid in db:
        all_ffields.append(polar_to_complex_conv(pyramid["result"]["far_fields"]))
        total_time += pyramid["result"]["Elapsed time (min)"]


    #Withdraw the E,H fields from polarization x,y,z
    #every triple is actually one dipole emission, but 1 pyramid emits from x, 1 from y, 1 from z, so they need to be added together, here called a "3-group"
    #this loop will go through the number of pyramids, but each pyramid will be selected in the inner loop

    #initialize lists for the loop below

    #stores the poyning vecors
    poynting_total_field =[0]*nfreq
    poynting_total_field=[poynting_total_field]*ff_pts
    #stores the total summed up flux
    total_flux = [0]*nfreq
    #stores the far field for one 3 group
    ff_flux = [0]*nfreq
    ff_flux_int = [0]*nfreq
    ff_flux_int2 = []
    total_flux_int = [0]*nfreq
    total_flux_int2 = []

    dipole_positions = []

    r=2*math.pow(db[0]["pyramid"]["pyramid_height"],2)*db[0]["pyramid"]["frequency_center"]*2*10
    S = 2*math.pi*pow(r,2)*(1-math.cos(theta))/range_npts
    print('Initializing quantum well calculations, number of 3-group is:',int(number_of_pyramids/3))
    print('Time to perform the simulations was:',round(total_time,0),' (min)')
    print('r,theta,range_npts',r,theta,range_npts)
    print('S',S)


    #save the initial far-field coordinates. The assumption is now that all the 3-groups have the same far-field sampling coordinates!
    initial_ff_coords = []
    far_field = db[5]["result"]["far_fields"]
    print('ff',len(far_field))
    for n in range(range_npts):
        ff_sampling_coord = far_field[n]["pos"]
        initial_ff_coords.append(ff_sampling_coord)
    print('Number of sampling coordinates is:')
    print(len(initial_ff_coords))
    #print(initial_ff_coords)
    #far-field coordinates for all 3-groups after rotation
    poynting_field_coords = []

    #rotation integers for all 3-groups
    rotation_integers = []

    #poynting field values for rotated coords
    poynting_total_field_rotated = [[0]*nfreq]*ff_pts

    #Initialize poynting field coordinates
    #Loop through each "3-group"
    for i in range(0,int(number_of_pyramids),3):

        
        #select dipole position for corresponding 3-group 
        current_dipole_position = db[i]["pyramid"]["source_position"]

        #randomize rotation (that is, randomize wall that dipole is located on)
        rotation_for_3g = randint(0,5)
        rotation_integers.append(rotation_for_3g)
        #rotation is counter-clockwise

        #rotation the far-field coordinates for the corresponding 3-group
        poynting_field_coords.append(rotate_coordinate_list(initial_ff_coords,rotation_for_3g))

        #save current dipole position. Current dipole position and the rotation integer can recreate the rotated dipole position.
        dipole_positions.append(current_dipole_position)

        #select position and field from all_ffields from pyramid with dipole x,y,z respectively.
        ff_x_dipole = all_ffields[i+0]
        ff_y_dipole = all_ffields[i+1]
        ff_z_dipole = all_ffields[i+2]

        #select only the field values at each far-field point
        fval_x_dipole = [d['field'] for d in ff_x_dipole]
        fval_y_dipole = [d['field'] for d in ff_y_dipole]
        fval_z_dipole = [d['field'] for d in ff_z_dipole]

        #randomize polarization for the 3-groups dipole
        pol_range = {"from": -1, "to": 1}
        wx,wy,wz = draw_uniform(pol_range,pol_range,pol_range)

        #Project the polarization down in the plane, that is, the pyramid wall
        wx_plane,wy_plane,wz_plane = PolarizeInPlane((wx,wy,wz),db[i]["pyramid"]["pyramid_height"],db[i]["pyramid"]["pyramid_width"])
       
        #Linear combine the 3-groups E,H fields 
        summed_ff_3g = linear_combine_fields(fval_x_dipole,fval_y_dipole,fval_z_dipole,wx_plane,wy_plane,wz_plane,ff_pts)

        #Calculate the resulting poynting field from the 3 group
        poynting_field_3g = calculate_poynting_field(summed_ff_3g,ff_pts,nfreq)
        #add the poynting field from the 3-group to the total field
        poynting_total_field = add_poynting_fields(poynting_total_field,poynting_field_3g, ff_pts)

        #Save a seperate poynting total field where the dipoles and the corresponding poynting field have rotated with n*60 degrees
        poynting_field_3g_rotated = rotate_list2(poynting_field_3g,npts,rotation_for_3g)
 
        poynting_total_field_rotated = add_poynting_fields(poynting_total_field_rotated,poynting_field_3g_rotated, ff_pts)

        #Add the poynting vector values 

        #ff_flux internal is set to 0 for each new three group. But the total quantum well energy values are added to the internal list each iteration
        ff_flux_internal = [0]*nfreq
        for n in range(ff_pts):
            for k in range(nfreq):
                ff_flux_internal[k] += S*poynting_total_field[n][k]

        #Due to the way lists are handle in python I can not save total_flux to a list so I need to create an internal list to save total_flux for each iteration
        total_flux_internal=[0]*nfreq
        for k in range(nfreq):
            total_flux[k] += db[i+0]["result"]["total_flux"][k]*wx_plane*wx_plane+db[i+1]["result"]["total_flux"][k]*wy_plane*wy_plane+db[i+2]["result"]["total_flux"][k]*wz_plane*wz_plane
            total_flux_internal[k] += total_flux[k]

        ff_flux_int2.append(ff_flux_internal)
        total_flux_int2.append(total_flux_internal)
 
        freq=2

    "calculate ff_flux for rotated poynting field"
    ff_flux_tot_rot = [0]*nfreq
    for n in range(ff_pts):
        for k in range(nfreq):
            ff_flux_tot_rot[k] += S*poynting_total_field_rotated[n][k]

    print('ff flux tot rot:',ff_flux_tot_rot)

    for n in range(ff_pts):
        for k in range(nfreq):
            ff_flux[k] += S*poynting_total_field[n][k]

    for k in range(nfreq):
        print('final freq',k,'ff_angle',round(ff_flux[k],4),'tot',round(total_flux[k],4),'ratio',round(100*ff_flux[k]/total_flux[k],2),'%')

    #Plot LEE convergence
    plot_LEE_convergence(ff_flux_int2,total_flux_int2)

    #Plot poynting coordinates
    plot_poynting_coordinates(initial_ff_coords)

    pyramid_height = db[0]["pyramid"]["pyramid_height"]
    inner_pyramid_height = pyramid_height - 0.1
    simulation_ratio = eval(db[0]["simulate"]["simulation_ratio"])
    substrate_ratio = eval(db[0]["simulate"]["substrate_ratio"])
   

    #PLOTTER FUNCTIONS
    #dipole plot
    plot_dipoles_on_pyramids(dipole_positions,rotation_integers,pyramid_height,simulation_ratio,substrate_ratio)

    #field freq to plot
    ff_pts_rot = []
    for n in range(ff_pts):
        ff_pts_rot.append(poynting_total_field_rotated[n][freq])
    ff_pts_norm = []
    for n in range(ff_pts):
        ff_pts_norm.append(poynting_total_field[n][freq])
    max_val2 = max(ff_pts_norm)

   # for n in range(ff_pts):
    #    ff_pts_norm[n] = ff_pts_norm[n]/max_val
    #print(ff_pts_norm)

    plot_farfield(initial_ff_coords,ff_pts_rot)
    plot_farfield(initial_ff_coords,ff_pts_norm)
    #plt.show()

    plt.hist(ff_pts_norm,100)
    #plt.show()
    #ax4 = plt.axes(projection='3d')
    #ax4.plot_trisurf(x,y, ff_pts_norm2,cmap='viridis', edgecolor='none')
    #plt.show()
    plt.hist(ff_pts_norm2,100,color='orange')
    plt.show()
   # plt.plot(x,z[24])
   # plt.xlabel('x-coordinates')
    #plt.ylabel('Normalized flux (y=0)')
    #print('z24',z[24])
    

 
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 :
        print(sys.argv)
        print([arg.strip() for arg in sys.argv[1:]])
        Qwell_wrapper(sys.argv[1],sys.argv[2])
    else:
        print("Specify database to run and the number of dipoles to use")
        exit(0)





