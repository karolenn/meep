from utils.api import *
from utils.functions import *
import matplotlib.pyplot as plt

#freqs [1.0, 1.2222222222222223, 1.4444444444444444, 1.6666666666666667, 1.8888888888888888, 2.111111111111111, 2.3333333333333335, 2.5555555555555554, 2.7777777777777777, 3.0]

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
    print('x',datax)
    print('y',datay)
    datay=sum(datay,[])
    print(datay)



if __name__ == "__main__":
    import sys
    if (len(sys.argv)) == 4:
        plotter(sys.argv[1],sys.argv[2],sys.argv[3])
    else:
        print("Not enough arguments")
        exit(0)
