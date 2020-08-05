from functionality.api import read, db_to_array, polar_to_complex_conv, complex_to_polar_conv
from functionality.functions import myPoyntingFlux,PolarizeInPlane,linear_combine_fields,calculate_poynting_field,add_poynting_fields
import matplotlib.pyplot as plt
import numpy as np
import math as math
from random import uniform
    #keeps track on number of emission positions and dipole arrangement per position


def Qwell_wrapper(sim_name,number_of_dipoles):

    db = read("db/initial_results/{}.json".format(sim_name))
    if db == None:
        print("could not open db/initial_results/{}.json".format(sim_name))
        exit(0)

    #TODO:verify that conversion from complex->polar->complex works
    all_ffields = []

    #Withdraw all the far fields from all the pyramids in active db
    for pyramid in db:
        all_ffields.append(polar_to_complex_conv(pyramid["result"]["far_fields"]))
    number_of_pyramids=len(all_ffields)-30*0
    ff_pts = db[0]["simulate"]["ff_pts"]
    theta = math.pi/db[0]["simulate"]["ff_angle"]
    if theta == math.pi/6:
        npts = ff_pts*3
    else:
        npts = ff_pts
    range_npts=int((theta/math.pi)*npts) 
    ff_pts = range_npts
    nfreq=db[0]["pyramid"]["number_of_freqs"]
    print('nr of pyr',number_of_pyramids)
    print('number of ff pts',len(all_ffields[0]))
    print('nr of freqs',nfreq)

    #Withdraw the E,H fields from polarization x,y,z
    #every triple is actually one dipole emission, but 1 pyramid emits from x, 1 from y, 1 from z, so they need to be added together, here called a "3-group"
    #this loop will go through the number of pyramids, but each pyramid will be selected in the inner loop
    poynting_total_field =[0]*nfreq
    poynting_total_field=[poynting_total_field]*ff_pts
    total_flux = [0]*nfreq
    ff_flux = [0]*nfreq
    ff_flux_int = [0]*nfreq
    ff_flux_int2 = []
    total_flux_int = [0]*nfreq
    total_flux_int2 = []
    

    r=2*math.pow(db[0]["pyramid"]["pyramid_height"],2)*db[0]["pyramid"]["frequency_center"]*2*10
    S = 2*math.pi*pow(r,2)*(1-math.cos(theta))/range_npts
    print('r,theta,range_npts',r,theta,range_npts)
    print('S',S)
    #Loop through each "3-group"
    for i in range(0,number_of_pyramids,3):
        #select position and field from all_ffields from pyramid with dipole x,y,z respectively.
        ff_x_dipole = all_ffields[i+0]
        ff_y_dipole = all_ffields[i+1]
        ff_z_dipole = all_ffields[i+2]
        #select only the field values at each far-field point
        fval_x_dipole = [d['field'] for d in ff_x_dipole]
        fval_y_dipole = [d['field'] for d in ff_y_dipole]
        fval_z_dipole = [d['field'] for d in ff_z_dipole]
        #randomize polarization for the 3-groups dipole
        wx= uniform(-1,1)
        wy= uniform(-1,1)
        wz= uniform(-1,1)
        #Project the polarization down in the plane, that is, the pyramid wall

        plane_pol = PolarizeInPlane((wx,wy,wz),db[i]["pyramid"]["pyramid_height"],db[i]["pyramid"]["pyramid_width"])
        
        wx_plane = plane_pol[0]
        wy_plane = plane_pol[1]
        wz_plane = plane_pol[2]

        #Linear combine the 3-groups E,H fields 

        summed_ff = linear_combine_fields(fval_x_dipole,fval_y_dipole,fval_z_dipole,wx_plane,wy_plane,wz_plane,ff_pts)

        #Calculate the resulting poynting field from the 3 group
        poynting_field = calculate_poynting_field(summed_ff,ff_pts,nfreq)

        #add the poynting field from the 3-group to the total field
        poynting_total_field = add_poynting_fields(poynting_total_field,poynting_field, ff_pts)

        #Add the poynting vector values 
        ff_flux_int = [0]*nfreq
        for n in range(ff_pts):
            for k in range(nfreq):
                ff_flux_int[k] += S*poynting_total_field[n][k]

       # for k in range(nfreq):
         #   for n in range(ff_pts):
        #        ff_flux_int[k] += S*poynting_total_field[n][k]

       # if i == 0:
        #    print(poynting_field)
        #    print(poynting_total_field)
        #Due to the way lists are handle in python I can not save total_flux to a list so I need to create an internal list to save total_flux for each iteration
        total_flux_int=[0]*nfreq
        for k in range(nfreq):
            total_flux[k] += db[i+0]["result"]["total_flux"][k]*wx_plane*wx_plane+db[i+1]["result"]["total_flux"][k]*wy_plane*wy_plane+db[i+2]["result"]["total_flux"][k]*wz_plane*wz_plane
            total_flux_int[k] += total_flux[k]



        ff_flux_int2.append(ff_flux_int)
        total_flux_int2.append(total_flux_int)
 

        i_ = int(i/3)
        freq=2
        current_total_ratio = round(ff_flux_int2[i_][freq]/total_flux_int2[i_][freq],4)
        print(f"pyramid group {i_}, result: total flux: {round(total_flux_int2[i_][freq],4)}, far field: {round(ff_flux_int2[i_][freq],6)}, ratio: {current_total_ratio}")
       # print('ff flux int',ff_flux_int)
      #  print('pyr flux',db[i+0]["result"]["total_flux"][freq]+db[i+1]["result"]["total_flux"][freq]+db[i+2]["result"]["total_flux"][freq])
       # print('ff int',ff_flux_int[freq])
        #print(db[i+0]["result"]["ff_at_angle"][freq],db[i+1]["result"]["ff_at_angle"][freq],db[i+2]["result"]["ff_at_angle"][freq])
     #   print(db[i+0]["result"]["total_flux"][freq],db[i+1]["result"]["total_flux"][freq],db[i+2]["result"]["total_flux"][freq])
     #   print('pyr ff',db[i+0]["result"]["ff_at_angle"][freq]+db[i+1]["result"]["ff_at_angle"][freq]+db[i+2]["result"]["ff_at_angle"][freq])
       # print('3-group flux',db[i+0]["result"]["total_flux"][k]*wx_plane*wx_plane+db[i+1]["result"]["total_flux"][k]*wy_plane*wy_plane+db[i+2]["result"]["total_flux"][k]*wz_plane*wz_plane)
        #print(i_,db[i_]["pyramid"]["source_direction"],db[i_]["result"]["ff_at_angle"][0],db[i_]["result"]["total_flux"][0],db[i_]["result"]["flux_ratio"][0],'tot ff',ff_flux_int2[i_][freq],'tot flux',total_flux_int2[i_][freq],'totratio',ff_flux_int2[i_][freq]/total_flux_int2[i_][freq])
     #   print('tot ff',ff_flux_int2[i_][freq],'tot flux',total_flux_int2[i_][freq],'totratio',ff_flux_int2[i_][freq]/total_flux_int2[i_][freq])

    "TEST CODE FOR VERIFICATION"
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
       # ratio_4.append(100*ff_flux_int2[n][3]/total_flux_int2[n][3])
      #  ratio_5.append(100*ff_flux_int2[n][4]/total_flux_int2[n][4])
        index.append(n)


    plt.plot(index,ratio_1,color='red',label='714 nm')
    plt.plot(index,ratio_2,color='g',label='500 nm')#588 orange
    plt.plot(index,ratio_3,color='m',label='385 nm')#500 g
   # plt.plot(index,ratio_4,color='b',label='435 nm')
   # plt.plot(index,ratio_5,color='m',label=' 385 nm')
    plt.legend(loc='best')
    plt.grid()
    plt.ylabel('LEE (%)')
    plt.xlabel('Number of dipoles')
    plt.show()




   #     withdraw E,H fields from ex,ey,ez (I also need to be able to scale these)
   #     calculate coherent field and Poynting field (calculate poynting scalar field )
   #     add poynting scalar field to total field
    #    repeat
    #no, unnessecary long sequence, fields are already in all_ffields. What do I REALLY wanna do here?



        



if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 :
        print(sys.argv)
        print([arg.strip() for arg in sys.argv[1:]])
        Qwell_wrapper(sys.argv[1],sys.argv[2])
    else:
        print("Specify database to run and the number of dipoles to use")
        exit(0)





