import json

# Loading the file and returns it if found else returns None
def read(db):
    try:
        with open (db, encoding='utf-8') as f:
            data_json = json.load(f)
            return data_json
    except OSError as err:
        print(err)
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
    
    write(db, data)
        

def sim_to_json(config,result):
    flux_tot_out, P_tot_ff, flux_tot_ff_ratio = result
    result = {"total_flux":flux_tot_out , "ff_at_angle":P_tot_ff , "flux_ratio":flux_tot_ff_ratio }
    config["result"] = result
    return config

def opt_to_json(opt):
    return opt

def json_to_imp(json_spec):
    return json_spec