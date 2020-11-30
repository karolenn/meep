from functionality.api import read, db_to_array, polar_to_complex_conv, complex_to_polar_conv
from functionality.functions import myPoyntingFlux,PolarizeInPlane, draw_uniform
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
    number_of_pyramids = 2
    ff_pts = db[0]["simulate"]["ff_pts"]
    theta = math.pi/db[0]["simulate"]["ff_angle"]
    #TODO: Redesign this ugly hack in the fibbonacci sampling algorithm
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
    for pyramid in db:
        all_ffields.append(polar_to_complex_conv(pyramid["result"]["far_fields"]))


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
    dipole_positions_rotated = []

    #
    all_field_pos = []

    r=2*math.pow(db[0]["pyramid"]["pyramid_height"],2)*db[0]["pyramid"]["frequency_center"]*2*10
    S = 2*math.pi*pow(r,2)*(1-math.cos(theta))/range_npts
    print('Initializing quantum well calculations, number of 3-group is:',number_of_pyramids/3)
    print('r,theta,range_npts',r,theta,range_npts)
    print('S',S)


    #save the initial far-field coordinates. The assumption is now that all the 3-groups have the same far-field sampling coordinates!
    initial_ff_coords = []
    far_field = db[0]["result"]["far_fields"]
    for n in range(ff_pts):
        ff_sampling_coord = far_field[n]["pos"]
        initial_ff_coords.append(ff_sampling_coord)

    #far-field coordinates for all 3-groups after rotation
    poynting_field_coords = []

    #rotation integers for all 3-groups
    rotation_integers = []

    #poynting field values for rotated coords
    poynting_total_field_rotated = []

    #Initialize poynting field coordinates
    poynting_total_field_rotated2 = initialize_poynting_far_field_rotated(ff_pts,nfreq)
    print(len(poynting_total_field))
    print(len(poynting_total_field_rotated2[0]))
    print(len(poynting_total_field_rotated2[1]))
    print(len(poynting_total_field_rotated2[2]))
    print(len(poynting_total_field_rotated2[3]))
    print(len(poynting_total_field_rotated2[4]))
    print(len(poynting_total_field_rotated2[5]))
    print(len(poynting_total_field_rotated2[5][0]))
    #Loop through each "3-group"
    for i in range(0,int(number_of_pyramids),3):

        
        #select dipole position for corresponding 3-group 
        current_dipole_position = db[i]["pyramid"]["source_position"]

        #randomize rotation (that is, randomize wall that dipole is located on)
        rotation_for_3g = randint(0,5)
        rotation_integers.append(rotation_for_3g)

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
        poynting_total_field_rotated.append(poynting_field_3g)

        poynting_total_field_rotated2[rotation_for_3g] = add_poynting_fields(poynting_total_field_rotated2[rotation_for_3g],poynting_field_3g,ff_pts)
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

        #CALCULATE 3-group far field (ONLY FOR TESTING PURPOSES)
        ff_flux_3group = [0]*nfreq
        for n in range(ff_pts):
            for k in range(nfreq):
                ff_flux_3group[k] += S*poynting_field_3g[n][k]

        #save the far-field coordinates

        ff_coords = []


        ff_flux_int2.append(ff_flux_internal)
        total_flux_int2.append(total_flux_internal)
 

        i_ = int(i/3)
        freq=2
        current_total_ratio = round(ff_flux_int2[i_][freq]/total_flux_int2[i_][freq],4)

        #print if the change from one iteration to the next changes by more than 20% 
        if abs((current_total_ratio - ff_flux_int2[i_-1][freq]/total_flux_int2[i_ - 1][freq])/current_total_ratio) > 0.10:
            print('huge diff detected:')
            print('dipole wieghts',wx_plane,wy_plane,wz_plane)
            print('index:',i_)
            print('total flux internal:',total_flux_int2[i_][freq])
            print('far field flux internal:',ff_flux_int2[i_][freq])
            print('current total ratio:',current_total_ratio)
            print('total flux internal prev:',total_flux_int2[i_-1][freq])
            print('far field flux internal prev:',ff_flux_int2[i_-1][freq])
            print('----')
            print('this 3-groups total flux and far field (for a given freq)')
            print('3g tot flux:',db[i+0]["result"]["total_flux"][freq]*wx_plane*wx_plane+db[i+1]["result"]["total_flux"][freq]*wy_plane*wy_plane+db[i+2]["result"]["total_flux"][freq]*wz_plane*wz_plane)
            print('3g ff:',ff_flux_3group[freq])
            print('LEE:',ff_flux_3group[freq]/(db[i+0]["result"]["total_flux"][freq]*wx_plane*wx_plane+db[i+1]["result"]["total_flux"][freq]*wy_plane*wy_plane+db[i+2]["result"]["total_flux"][freq]*wz_plane*wz_plane))
      #  print(f"pyramid group {i_}, result: total flux: {round(total_flux_int2[i_][freq],4)}, far field: {round(ff_flux_int2[i_][freq],6)}, ratio: {current_total_ratio}")

    "calculate ff_flux for rotated poynting field"
    ff_flux_tot_rot = [0]*nfreq
    for j in range(int(number_of_pyramids/3)):
        for n in range(ff_pts):
            for k in range(nfreq):
                ff_flux_tot_rot[k] += S*poynting_total_field_rotated[j][n][k]
    #for n in range(ff_pts):

    print('ff flux tot rot:',ff_flux_tot_rot)
    "TEST CODE FOR VERIFICATION"
  #  print(all_field_pos)
    print('ff db sum')
    sum_ff=[0]*nfreq
    sum_tot=[0]*nfreq
    for pyramid in db:
        for k in range(nfreq):
            sum_ff[k] += pyramid["result"]["ff_at_angle"][k]
            sum_tot[k] += pyramid["result"]["total_flux"][k]
    print('db um',sum_ff,sum_tot)

    for n in range(ff_pts):
        for k in range(nfreq):
            ff_flux[k] += S*poynting_total_field[n][k]


    for k in range(nfreq):
        print('final freq',k,'ff_angle',round(ff_flux[k],4),'tot',round(total_flux[k],4),'ratio',round(100*ff_flux[k]/total_flux[k],2),'%')

    #plot poynting coordinates
    x,y,z = create_far_field_coords(initial_ff_coords)
    print(len(x))
    ax2 = plt.axes(projection='3d')
    ax2.scatter(x,y,z)
    X = []
    Y = []
    Z = []
    for k in range(len(initial_ff_coords)):
        X.append(initial_ff_coords[k][0])
        Y.append(initial_ff_coords[k][1])
        Z.append(initial_ff_coords[k][2])
    ax2.scatter(X,Y,Z)
    plt.show()

    ff_values = []
    print('rang',len(poynting_total_field_rotated2))
    print('field',poynting_total_field_rotated2)
    for i in range(6):
        for n in range(len(poynting_total_field_rotated2[i])):
            val = poynting_total_field_rotated2[i][n][freq]
            ff_values.append(val)
    print('ff al',ff_values)
    print('lenn',len(ff_values))
    #plot emissi,on lobe
    ax3 = plt.axes(projection='3d')
    #ax3.plot_trisurf(x, y, ff_values,cmap='viridis', edgecolor='none')


    X = []
    Y = []
    Z = []
    for n in range(len(x)):
        X.append(x[n]*ff_values[n])
        Y.append(y[n]*ff_values[n])
        Z.append(z[n]*ff_values[n])

    ax3.plot_trisurf(X,Y,Z,cmap='jet',edgecolor='none')
    
    ax3.set_zlabel(r'$ flux')
    ax3.set_title(r'$\lambda = 500 nm$')
    ax3.set_xlabel('x-coordinates')
    ax3.set_ylabel('y-coordinates')
    plt.show()


    ff_1 = []
    ff_2 = []
    ff_3 = []
    tot_1 = []
    tot_2 = []
    tot_3 = []
    ratio_1 = []
    ratio_2 = []
    ratio_3 = []
    ratio_4 = []

    ratio_5 = []
    index = []
    for n in range(len(ff_flux_int2)):
        ratio_1.append(100*ff_flux_int2[n][0]/total_flux_int2[n][0])
        ratio_2.append(100*ff_flux_int2[n][1]/total_flux_int2[n][1])
        ratio_3.append(100*ff_flux_int2[n][2]/total_flux_int2[n][2])
        ratio_4.append(100*ff_flux_int2[n][3]/total_flux_int2[n][3])
        ratio_5.append(100*ff_flux_int2[n][4]/total_flux_int2[n][4])
        index.append(n)


    #plt.plot(index,ratio_1,color='red',label='714 nm')
    plt.plot(index,ratio_2,color='orange',label='588 nm')#588 orange
    plt.plot(index,ratio_3,color='g',label='500 nm')#500 g
    plt.plot(index,ratio_4,color='b',label='435 nm')
    #plt.plot(index,ratio_5,color='m',label=' 385 nm')
    plt.legend(loc='best')
    plt.grid()
    plt.ylabel('LEE (%)')
    plt.xlabel('Number of dipoles')
    plt.show()


    pyramid_height = db[0]["pyramid"]["pyramid_height"]
    inner_pyramid_height = pyramid_height - 0.1
    simulation_ratio = 6/5
    subsrate_ratio = 1/10
    sh = subsrate_ratio*pyramid_height
    sz=pyramid_height*simulation_ratio

    #PLOTTER FUNCTIONS
    #dipole plot
    ax = plt.axes(projection='3d')
    print('dipol pos list length',len(dipole_positions))
    x_spos,y_spos,z_spos = zip(*dipole_positions)
    X_ = []
    Y_ = []
    Z_ = []
    xspos = []
    yspos = []
    zspos = []
    for n in range(len(x_spos)):
        #TODO: Create function that does this mapping
        x = (inner_pyramid_height*z_spos[n]*math.cos(math.pi/6))/math.tan(62*math.pi/180)
        y = y_spos[n]*math.tan(math.pi/6)*(inner_pyramid_height*z_spos[n]*math.cos(math.pi/6))/math.tan(62*math.pi/180)
        z = sz/2-sh-inner_pyramid_height+inner_pyramid_height*z_spos[n]

        #rotate dipole position
        x,y,z = rotate_coordinate(x,y,z,rotation_integers[n])

        xspos.append(x)
        yspos.append(y)
        zspos.append(z)

    ax.scatter(xspos,yspos,zspos)
    plt.show()


    #field freq to plot
    ff_pts_norm = []
    for i in range(int(number_of_pyramids/3)):
        for n in range(ff_pts):
            ff_pts_norm.append(poynting_total_field_rotated[i][n][freq])
    max_val = max(ff_pts_norm)

   # for n in range(ff_pts):
    #    ff_pts_norm[n] = ff_pts_norm[n]/max_val
    #print(ff_pts_norm)

    ff_coords = []
    #""   for k in range(len(all_ffields[0])):
    #      ff_coords.append(all_ffields[0][k]["pos"])
    print('len',len(poynting_field_coords))
    print('len2',len(poynting_field_coords[1]))
   # print(poynting_field_coords[1])
    for i in range(int(number_of_pyramids/3)):
        for n in range(ff_pts):
            ff_coords.append(poynting_field_coords[i][n])

    #print(ff_coords)
    x_coord = []
    phi = []
    y_coord = []
    theta = []
  # print('ff_coords',ff_coords)
   # print('ff',ff_pts_norm)
    for k in range(len(ff_coords)):
        x = ff_coords[k][0]
        y = ff_coords[k][1]
        if y == 0:
            y = y + 1e-9
        z = ff_coords[k][2]
        x_coord.append(ff_coords[k][0])
        y_coord.append(ff_coords[k][1])


        phi_ = math.atan(x/y)
        phi.append(phi_)
        theta_ = math.atan(math.sqrt(x**2+y**2)/z)
        theta.append(theta_)


    ax.scatter(x_coord,y_coord,ff_pts_norm)
    plt.show()
    from scipy.interpolate import Rbf

  #  RBF = Rbf(x_coord,y_coord,ff_pts_norm,epsilon = 2)

    x = np.linspace(min(x_coord),max(x_coord))
    y = np.linspace(min(y_coord),max(y_coord))


    #generate a list of all possible poynting field cooordinates
    

   # rbf_data = RBF(x,y)
    print('ff pts norm',len(ff_pts_norm))
    print('stats')
    import statistics as stats
    print(stats.mean(ff_pts_norm))
    print(stats.median(ff_pts_norm))
    print(stats.stdev(ff_pts_norm))
    #select cross sections
    x_cr = []
    y_cr = []
    ff_pts_cr = []
    for k in range(len(ff_coords)):
        if abs(ff_coords[k][0]) < 20:
            x_cr.append(ff_coords[k][0])
            ff_pts_cr.append(ff_pts_norm[k])

 
    x_cr,ff_pts_cr = zip(*sorted(zip(x_cr,ff_pts_cr)))

    print('xcord len',len(x_coord))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x_coord, y_coord, ff_pts_norm,cmap='viridis', edgecolor='none')
    ax.set_zlabel(r'$ flux')
    ax.set_title(r'$\lambda = 500 nm$')
    ax.set_xlabel('x-coordinates')
    ax.set_ylabel('y-coordinates')
  #  X, Y = np.meshgrid(x, y)
   # z = RBF(X, Y)

    plt.show()

    plt.hist(ff_pts_norm,100)
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





