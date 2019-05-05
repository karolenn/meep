from utils.api import *
from utils.functions import *
import matplotlib.pyplot as plt


def plotter(db,x,y):
    db = read("db/initial_results/{}.json".format(db))
    if db == None:
        print("could not open db/initial_results/{}.json".format(db))
        exit(0)
    datax = db_to_array(db,"pyramid",x)
    datay = db_to_array(db,"result",y)
    plt.title("plotting:"+' '+'xaxis:'+str(x)+' '+'yaxis:'+str(y))
    plt.plot(datax,datay)
    plt.show()
    print(datay)



if __name__ == "__main__":
    import sys
    if (len(sys.argv)) == 4:
        plotter(sys.argv[1],sys.argv[2],sys.argv[3])
    else:
        print("Not enough arguments")
        exit(0)
