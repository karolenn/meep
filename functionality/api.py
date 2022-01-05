import json
import numpy as np
import itertools

#from functionality.functions import complex_to_polar_conv
# Loading the file and returns it if found else returns None
def read(db):
    try:
        with open (db, encoding='utf-8') as f:
            data_json = json.load(f)
            return data_json
    except OSError as err:
        print('unable to load using json.load,error:',err)
        return None

def write(db, data):
    try:
        with open(db, 'w') as outfile:
            json.dump(data, outfile)
    except OSError as err:
        print('unable to write using json.dump,error:',err)
        return False
    return True

def write_result(db, info):
    data = read(db)
    if data != None:
        data.append(info)
    else:
        data = [info]
    write(db, data)

#extract data from the first pyramid in a db.json object
def extract_data_from_db(db):
    db = read("../db/{}.json".format(db))
    ff = db[0]["result"]["fields"]
    ff = polar_to_complex_conv(ff)
    npts = len(ff)
    nfreq = db[0]["pyramid"]["number_of_freqs"]
    ff_angle = db[0]["simulate"]["ff_angle"]
    ph = db[0]["pyramid"]["pyramid_height"]
    fcen = db[0]["pyramid"]["frequency_center"]
    return ff, npts, nfreq, ff_angle, ph, fcen

#given a list [[2,3,4],[6,7]] -> [2,3,4,6,7]
def unpack_list(list_tmp):
	list_flat = list(itertools.chain(*list_tmp))
	return list_flat

#extract field values from a ff object
#[{'pos':[x,y,z],'field':[Ex_f1,Ey_f1,...,Hz_fn]},,{..},{'pos':[xn,yn,zn],'field':[Exn_f1,..,Hzn_fn]}] -> [[Ex1_f1,,..,Hz1_fn],..,[Exn_f1,..,Hzn_fn]]
def return_field_values_from_ff(ff):
    npts = len(ff)
    ff_values = []
    for n in range(npts):
        ff_values.append(ff[n]["field"])
    return ff_values

#extract field values from a ff object
#[{'pos':[x,y,z],'field':[Ex_f1,Ey_f1,...,Hz_fn]},,{..},{'pos':[xn,yn,zn],'field':[Exn_f1,..,Hzn_fn]}] -> [[x,y,z],..,[xn,yn,zn]]
def return_position_values_from_ff(ff):
    npts = len(ff)
    ff_pos = []
    for n in range(npts):
        ff_pos.append(ff[n]["pos"])
    return ff_pos  


#Convert far-fields form complex numbers to polar "numbers" because JSON sucks (can't store complex numbers)
#TODO:this should be stored in functions.py but importing from there would create circular importing
#list dict is in the form 
def complex_to_polar_conv(list_dict):
    for k in range(len(list_dict)):
        for j in range(len(list_dict[0]["field"])):
            #numpy maps angle to (-pi,pi]
            list_dict[k]["field"][j] = (np.abs(list_dict[k]["field"][j]),np.angle(list_dict[k]["field"][j]))
    return list_dict

#convert polar numbers to complex numbers
#a+b*i=r*e^(i*angle)
def polar_to_complex_conv(list_dict):
    #withdraw the number of ff points
    for k in range(len(list_dict)):
        #withdraw the number of points times frequencies (?)
        for i in range(len(list_dict[0]["field"])):
            list_dict[k]["field"][i] = list_dict[k]["field"][i][0]*np.exp(1j*list_dict[k]["field"][i][1])
    return list_dict



#TODO: rewrite to be able to handle arbitrary results
def sim_to_json(config,result,output_ff =False, calculate_source_flux=False):
  #  len_results = len(result)
  #  result = dict(zip(result),zip(str(result)))
    config["result"] = result
    if output_ff:
        flux_tot_out, P_tot_ff, flux_tot_ff_ratio, fields, elapsed_time = result
        #JSON cant handle complex number so it is converted to polar coords. I want to save E,H fields so this seems to be a plausible workaround
        fields = complex_to_polar_conv(fields)
        result = {"total_flux":flux_tot_out , "ff_at_angle":P_tot_ff , "flux_ratio":flux_tot_ff_ratio , "far_fields":fields, "Elapsed time (min)":elapsed_time }
        config["result"] = result
    elif calculate_source_flux:
        flux_tot_out, source_flux_out, P_tot_ff, flux_tot_ff_ratio, elapsed_time = result
        result = {"total_flux":flux_tot_out , "source_flux": source_flux_out, "ff_at_angle":P_tot_ff , "flux_ratio":flux_tot_ff_ratio , "Elapsed time (min)":elapsed_time }
        config["result"] = result        
    else:
        flux_tot_out, P_tot_ff, flux_tot_ff_ratio, elapsed_time = result
        result = {"total_flux":flux_tot_out , "ff_at_angle":P_tot_ff , "flux_ratio":flux_tot_ff_ratio , "Elapsed time (min)":elapsed_time }
        config["result"] = result
    return config
#"Takes in flux ratio results, withdraws flux ratio above OR below for center frequency"
#"Function have two cases, flux ratio contains both above and below or only one of the cases. Controlled in the if case."
def process_results(sim_results,ff_calc, freqn="center"):
    #Withdraw ff_ratio above or under from results.
   # if float(sim_results[0][0]): #Flux ratio only contains result above / under
    new_results=[0]*len(sim_results)
    #print(sim_results)
    if len(sim_results[0])==1: 
        nfreqs=len(sim_results[0])
     #   if ff_calc == "Below":
      #      for n in range(len(sim_results)):
      #          sim_results[n]=sim_results[n][int(nfreqs/2)]
      #  else:
        for n in range(len(sim_results)):
            new_results[n]=sim_results[n][int(nfreqs/2)]
        return new_results
    else: #flux ratio contains both above and under ratios
        nfreqs=len(sim_results[0][0])
        if freqn=="center":
            freq=int(nfreqs/2)
            print('frequency:',freq)
        else:
            freq=freqn
            print('frequency:',freq)
        if ff_calc == "Below":
            for n in range(len(sim_results)):
                new_results[n]=sim_results[n][1][freq]
        else:
            for n in range(len(sim_results)):
                new_results[n]=sim_results[n][0][freq]
       # print('procc res2',sim_results)
        return new_results

#Returns total flux for a given frequency as a list
def return_total_flux(sim_results, freqn):
    new_results=[]
    for n in range(len(sim_results)):
        new_results.append(sim_results[n][freqn])
    return new_results

###Convert json result entries to python lists
###Usage: dataarray = db_to_array(db,pyramid,yourdata)"
def db_to_array(db,arg1,arg2):
    results=[]
    for result in db:
        results.append(result[arg1][str(arg2)])
    return results

