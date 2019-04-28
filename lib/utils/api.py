import json

# Loading the file and returns it if found else returns None
def read(db):
    try:
        with open (db, encoding='utf-8') as f:
            data_json = json.load(f)
            return data_json
    except OSError as err:
        print("unable to open")
        #print(err)
        return None

def write(db, data):
    try:
        with open(db, 'w') as outfile:
            json.dump(data, outfile)
    except OSError as err:
        print("unable to write")
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
    flux_tot_out, P_tot_ff, flux_tot_ff_ratio = result
    result = {"total_flux":flux_tot_out , "ff_at_angle":P_tot_ff , "flux_ratio":flux_tot_ff_ratio }
    config["result"] = result
    return config

def db_to_array(db,arg1,arg2):
   # print(db)
    #results = [[] for _ in range(4)]
    results=[]
    print(results)
    for result in db:
        results.append(result[arg1][str(arg2)])
        #print(result)
        #results[0].append(result["pyramid"]["source_position"])
        #results[1].append(result["pyramid"]["pyramid_height"])
        #results[2].append(result["pyramid"]["pyramid_width"])
        #results[3].append(result["result"]["flux_ratio"])
    print(results)
    return results

def opt_to_json(opt):
    raise "not implemented"

def json_to_imp(json_spec):
    raise "not implemented"