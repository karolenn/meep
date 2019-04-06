from utils.api import read, write

def main():
    db = read("db/data.json")
    for p in db["ppl"]:
        print(p["age"])
    
    print("db", db)
    db["ppl"].append({"name": "harald", "age": 4})
    write("db/data.json", db)





if __name__ == "__main__":
    main()