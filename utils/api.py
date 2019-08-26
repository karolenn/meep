import json

# Loading the file and returns it if found else returns None
def read(db):
    try:
        with open (db, encoding='utf-8') as f:
            data_json = json.load(f)
            return data_json
    except OSError as err:
        print('unable to load using json.load')
        return None

def write(db, data):
    try:
        with open(db, 'w') as outfile:
            json.dump(data, outfile)
    except OSError as err:
        print(err)
        return False
    return True

def write_result(db, info):
    data = read(db)
    if data != None:
        data.append(info)
    else:
        data = [info]

    
    write(db, data)
        

def sim_to_json(config,result):
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

"Usage: dataarray = db_to_array(db,pyramid,yourdata)"
def db_to_array(db,arg1,arg2):
   # print(db)
    #results = [[] for _ in range(4)]
    results=[]
    for result in db:
        results.append(result[arg1][str(arg2)])
        #print(result)
        #results[0].append(result["pyramid"]["source_position"])
        #results[1].append(result["pyramid"]["pyramid_height"])
        #results[2].append(result["pyramid"]["pyramid_width"])
        #results[3].append(result["result"]["flux_ratio"])
    return results

def opt_to_json(opt):
    return opt

def json_to_imp(json_spec):
    return json_spec
