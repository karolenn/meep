from functionality.api import read, db_to_array, polar_to_complex_conv
    #keeps track on number of emission positions and dipole arrangement per position

def linear_combine_fields(f1,f2,f3,wx,wy,wz):
    tmp = []
    for i in range(ff_pts):
        #sum up E,H field appropriately and keep track of freqs
    return tmp

def calculate_poynting_field(master_field,number_of_dipoles):
    tmp = []
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
    print('llll')
    print('nr of pyr',number_of_pyramids)
    print('number of ff pts',len(all_ffields[0]))
    #print(all_ffields)
    #Withdraw the E,H fields from polarization x,y,z
    #every triple is actually one dipole emission, but 1 pyramid emits from x, 1 from y, 1 from z, so they need to be added together
    #this loop will go through the number of pyramids, but each pyramid will be selected in the inner loop
    poynting_master_field = []
    print(all_ffields[0])
    print(all_ffields[3])
    for i in range(0,number_of_pyramids,3):
        #select position and field from all_ffields from pyramid with dipole x,y,z respectively.
        ff_x_dipole = all_ffields[i+0]
        ff_y_dipole = all_ffields[i+1]
        ff_z_dipole = all_ffields[i+2]
        #select only the field values
        fval_x_dipole = [d['field'] for d in ff_x_dipole]
        fval_y_dipole = [d['field'] for d in ff_y_dipole]
        fval_z_dipole = [d['field'] for d in ff_z_dipole]
        combined_ff = linear_combine_fields(fval_x_dipole,fval_y_dipole,fval_z_dipole,1,1,1)
        poynting_field = calculate_poynting_field(combined_ff,number_of_pyramids)
        poynting_master_field = linear_combine_fields(poynting_master_field,poynting_field,0,1,1,0)

    print(fval_x_dipole)
#possibly add the poynting field to the corresponding pyramid



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





