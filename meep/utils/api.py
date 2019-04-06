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