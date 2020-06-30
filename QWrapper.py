from functionality.api import read, db_to_array, polar_to_complex_conv
from functionality.functions import myPoyntingFlux
import numpy as np
import math as math
from random import uniform
    #keeps track on number of emission positions and dipole arrangement per position

#this function linearly adds E,H fields with weights wx, wy, wz which corresponds due to linearity to the pyramids polarization, so (wx=1,wy=0,wz=0) is x polarized dipole
#TODO:We might need to convert this to a faster method with increasing ffpts
def linear_combine_fields(f1,f2,f3,wx,wy,wz,ff_pts):
    tmp = []
    #f is a list of lists of length ff_pts
    for ffpt_i in range(ff_pts):
        tmp.append([wx*px + wy*py + wz*pz for px, py, pz in zip(f1[ffpt_i], f2[ffpt_i], f3[ffpt_i])])
    return tmp

def add_poynting_fields(P_ff1,P_ff2,ff_pts):
    tmp = []
    #print(P_ff1)
    for ffpt_i in range(ff_pts):
        tmp.append([a + b for a, b in zip(P_ff1[ffpt_i], P_ff2[ffpt_i])])
    return tmp



#Calculate the poynting scalar field for a 3-group of pyramids
#works for 1 pyramid
def calculate_poynting_field(summed_ff,number_of_pyramids,ff_pts,nfreq):
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
    ff_pts = len(all_ffields[0])
    nfreq=db[0]["pyramid"]["number_of_freqs"]
    print('nr of pyr',number_of_pyramids)
    print('number of ff pts',len(all_ffields[0]))
    print('nr of freqs',nfreq)

    #Withdraw the E,H fields from polarization x,y,z
    #every triple is actually one dipole emission, but 1 pyramid emits from x, 1 from y, 1 from z, so they need to be added together, here called a "3-group"
    #this loop will go through the number of pyramids, but each pyramid will be selected in the inner loop
    poynting_master_field =[0]*nfreq
    poynting_master_field=[poynting_master_field]*ff_pts
    total_flux = [0]*nfreq
    ff_flux = [0]*nfreq
    ff_flux_int = [0]*nfreq



    r=2*math.pow(db[0]["pyramid"]["pyramid_height"],2)*db[0]["pyramid"]["frequency_center"]*2*10
    theta = math.pi/db[0]["simulate"]["ff_angle"]
    if theta == math.pi/6:
        npts = ff_pts*3
    else:
        npts = ff_pts
    range_npts=int((theta/math.pi)*npts) 
    print(r,theta,range_npts)
    S = 2*math.pi*pow(r,2)*(1-math.cos(theta))/60
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
        #Linear combine the 3-groups E,H fields 
        summed_ff = linear_combine_fields(fval_x_dipole,fval_y_dipole,fval_z_dipole,wx,wy,wz,ff_pts)



        #print('summed_ff',summed_ff)
        #    poyn_field = calculate_poynting_field(summed_ff,number_of_pyramids,ff_pts,nfreq)
        #Calculate the resulting poynting field from the 3 group, and scale with number of pyramids 
        poynting_field = calculate_poynting_field(summed_ff,number_of_pyramids,ff_pts,nfreq)
      #  print('poynting field',poynting_field)
        poynting_master_field = add_poynting_fields(poynting_master_field,poynting_field, ff_pts)

        for n in range(ff_pts):
            for k in range(nfreq):
                ff_flux_int[k] += poynting_master_field[n][k]
        print(ff_flux_int)
        for k in range(nfreq):
            total_flux[k] += db[i+0]["result"]["total_flux"][k]*wx*wx+db[i+1]["result"]["total_flux"][k]*wy*wy+db[i+2]["result"]["total_flux"][k]*wz*wz
            print('pyramid,freq, ratio:',i,k,ff_flux_int[k]/total_flux[k])
          #  print(k,db[i+0]["result"]["total_flux"][k]*wx*wx)
          #  print(k,db[i+1]["result"]["total_flux"][k]*wy*wy)
          #  print(k,db[i+2]["result"]["total_flux"][k]*wz*wz)
        #print('pol',wx,wy,wz)
       # print('total flux',total_flux)
       # print('new pyramid',i)

   # print('poynting field fin',poynting_field)
       # print('master:',poynting_master_field)


    "TEST CODE FOR VERIFICATION"


    for n in range(ff_pts):
        for k in range(nfreq):
            ff_flux[k] += S*poynting_master_field[n][k]
    print('ff flux',ff_flux)
    print('tot flux',total_flux)
    for k in range(nfreq):
        print('final freq,ratio:',k,ff_flux[k]/total_flux[k])





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





