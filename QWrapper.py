from functionality.api import read, db_to_array, polar_to_complex_conv
from functionality.functions import myPoyntingFlux
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
def calculate_poynting_field(summed_ff,number_of_pyramids,ff_pts,nfreq):
    tmp = []
    ff_at_pt = []
    for ffpt_i in range(ff_pts):
        ff_at_pt = summed_ff[ffpt_i]
        print('ff_at_pt',ff_at_pt)
        poynting_vector_at_pt=[]
        i=0
        for freq in range(nfreq):
            Pr = myPoyntingFlux(ff_at_pt,i)
            poynting_vector_at_pt.append(Pr)
            print('vec',poynting_vector_at_pt)
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
    number_of_pyramids=len(all_ffields)
    ff_pts = len(all_ffields[0])
    nfreq=db[0]["pyramid"]["number_of_freqs"]
    print('nr of pyr',number_of_pyramids)
    print('number of ff pts',len(all_ffields[0]))

    #Withdraw the E,H fields from polarization x,y,z
    #every triple is actually one dipole emission, but 1 pyramid emits from x, 1 from y, 1 from z, so they need to be added together, here called a "3-group"
    #this loop will go through the number of pyramids, but each pyramid will be selected in the inner loop
    poynting_master_field =[0]*nfreq
    poynting_master_field=[poynting_master_field]*ff_pts

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
        #Linear combine the 3-groups E,H fields 
        summed_ff = linear_combine_fields(fval_x_dipole,fval_y_dipole,fval_z_dipole,1,1,1,ff_pts)
        #if i == 0:
        #    print('fvalx',fval_x_dipole)
        #    print('summed_ff',summed_ff)
        #Calculate the resulting poynting field from the 3 group, and scale with number of pyramids 
      #  poynting_field = calculate_poynting_field(summed_ff,number_of_pyramids,ff_pts,nfreq)
     #   print('poynting field',poynting_field)
     #   poynting_master_field = add_poynting_fields(poynting_master_field,poynting_field, ff_pts)


    print('master:',poynting_master_field)


    "TEST CODE FOR VERIFICATION"
    surface_Element = 3713.656582066039
    exff=all_ffields[0]
    #print('exff',exff)
    exfval = [d['field'] for d in exff]
    print('vals',exfval)
    print('---')
    expoyn = calculate_poynting_field(exfval,number_of_pyramids,ff_pts,nfreq)


    print('poyn',expoyn)








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





