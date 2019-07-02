from utils.api import *
from utils.functions import *
import matplotlib.pyplot as plt
import sys
import math
#freqs pml [1.0, 1.2222222222222223, 1.4444444444444444, 1.6666666666666667, 1.8888888888888888, 2.111111111111111, 2.3333333333333335, 2.5555555555555554, 2.7777777777777777, 3.0]
dpml_processing = True
time_processing = True
db1r=sys.argv[1]
db2r=sys.argv[2]
db3r=sys.argv[3]
#db = read("db/initial_results/{}.json".format(db))
#if db == None:
 #   print("could not open db/initial_results/{}.json".format(db))
  #  exit(0)
#datax = db_to_array(db,"pyramid","pyramid_height")
#datay = db_to_array(db,"result","flux_ratio")
#print(datay)
#print(datax)
if time_processing:
    db1 = read("db/initial_results/convtime1.json")   
    db2 = read("db/initial_results/convtime3.json")
    db3 = read("db/initial_results/convtime5.json")
    datax = db_to_array(db1,"pyramid","pyramid_height")
    datay1 = db_to_array(db1,"result","flux_ratio")
    datay2 = db_to_array(db2,"result","flux_ratio")
    datay3 = db_to_array(db3,"result","flux_ratio")
    ploty1=[]
    ploty2=[]
    ploty3=[]
    k=1 #freq index
    for n in range(len(datay1)):
        ploty1.append(datay1[n][k]) 
        ploty2.append(datay2[n][k])
        ploty3.append(datay3[n][k])
if dpml_processing:
    k=5 #frequency index 
    db1 = read("db/initial_results/{}.json".format(db1r))   
    db2 = read("db/initial_results/{}.json".format(db2r))
    db4 = read("db/initial_results/{}.json".format(db3r))
    X_Chosen = []
    for pyramid in db1:
        print('pyramid height',pyramid["pyramid"]["pyramid_height"],'pyramid width',pyramid["pyramid"]["pyramid_width"])
        if pyramid["pyramid"]["pyramid_height"]==1:
            X_Chosen.append(pyramid["result"]["flux_ratio"][k])
    print("X CHOSEN",X_Chosen)
    dataX = db_to_array(db1,"pyramid","pyramid_height")
    dataY=db_to_array(db1,"pyramid","pyramid_width")
    print('datax',dataX)
    datay1 = db_to_array(db1,"result","flux_ratio")
    datay2 = db_to_array(db2,"result","flux_ratio")
    datay4 = db_to_array(db4,"result","flux_ratio")
    ploty1=[]
    ploty2=[]
    ploty4=[]
    print('data',datay1,'len',len(datay1))
    print('data',datay2,'len',len(datay2))
    print('data',datay4,'len',len(datay4))

    for n in range(len(datay1)):
        ploty1.append(datay1[n][k]) 
        ploty2.append(datay2[n][k])
        ploty4.append(datay4[n][k])
    error=0
    for n in range(len(ploty1)):
        error+=abs(ploty1[n]-ploty4[n])
    print('error',error)
    print(ploty1)
    print(ploty2)
    print(ploty4)
    ax=plt.axes(projection='3d')
    ax.scatter(dataX,dataY,ploty1,marker='.',label='res 30')
    ax.scatter(dataX,dataY,ploty2,marker='^',label='res 60')
    ax.scatter(dataX,dataY,ploty4,marker='o',label='res 120')
    xi = np.linspace(min(dataX),max(dataX))
    yi = np.linspace(min(dataX),max(dataY))
    XI,YI=np.meshgrid(xi,yi)
    #ZI = RBF_Func(XI,YI)
    diff=[]
    for n in range(len(ploty4)):
        diff.append(ploty4[n]-ploty2[n])
    print(diff)
    #ax.plot_trisurf(dataX,dataY,diff,label='test',cmap=cm.jet)
    ax.set_xlabel('Pyramid height')
    ax.set_ylabel('Pyramid width')
    ax.set_zlabel('Flux ratio')
#plt.title('Flux ratio for 1 μ wide and high pyramid, varying source position, Y-polarized ')
    ax.view_init(elev=32., azim=37)
    plt.legend(loc='best')
    plt.show() 
    
    #plt.title("plotting:"+' '+'xaxis:'+str(x)+' '+'yaxis:'+str(y))
    #plt.plot(datax,datay)
    #plt.show()
    #print('x',datax)
    #print('y',datay)



#if __name__ == "__main__":
 #   import sys
  #  if (len(sys.argv)) == 4:
#        plotter(sys.argv[1],sys.argv[2],sys.argv[3])
#    else:
#        print("Not enough arguments")
 #       exit(0)
