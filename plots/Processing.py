from functionality.api import *
from functionality.functions import *
import matplotlib.pyplot as plt
import sys
import math

####SOME PROCESSING TO CREATE SOME NICE PLOTS FOR THE THESIS##################

#freqs pml [1.0, 1.2222222222222223, 1.4444444444444444, 1.6666666666666667, 1.8888888888888888, 2.111111111111111, 2.3333333333333335, 2.5555555555555554, 2.7777777777777777, 3.0]
dpml_processing = False
time_processing = False
res_process = True
av_time = False
#db1r=sys.argv[1]
#db2r=sys.argv[2]
#db3r=sys.argv[3]
#db = read("db/initial_results/{}.json".format(db))
#if db == None:
 #   print("could not open db/initial_results/{}.json".format(db))
  #  exit(0)
#datax = db_to_array(db,"pyramid","pyramid_height")
#datay = db_to_array(db,"result","flux_ratio")
#print(datay)
#print(datax)
if res_process:
    db1 = read("../db/initial_results/convlarge30.json")   
    db2 = read("../db/initial_results/convlarge60.json")
    db3 = read("../db/initial_results/convlarge120.json")  
    datay1 = db_to_array(db1,"result","flux_ratio")
    datay2 = db_to_array(db2,"result","flux_ratio")
    datay3 = db_to_array(db3,"result","flux_ratio")
    ph = db_to_array(db1,"pyramid","pyramid_height")
    pw = db_to_array(db1,"pyramid","pyramid_width")
    sp= db_to_array(db1,"pyramid","source_position")
    print(ph,pw,sp)
    for i in range(len(pw)):
        ph[i]=int(ph[i])
        pw[i]=int(pw[i])
    comb = list(zip(ph,pw,sp))
    print(comb)
    ploty1=[]
    ploty2=[]
    ploty3=[]
    #frequencies [1.0, 1.2222222, 1.444444444, 1.666666667, 1.888888, 2.1111111, 2.33335, 2.55555554, 2.777777777, 3.0]

    k=7
    for n in range(len(datay2)):
        ploty1.append(datay1[n][k]) 
        ploty2.append(datay2[n][k])
        ploty3.append(datay3[n][k])
    print('ff30=',ploty1)
    print('ff60=',ploty2)
    print('ff120=',ploty3)
    my_xticks = comb
    index = []
    for i in range(25):
        index.append(i)
    plt.xticks(index, my_xticks,fontsize=5,rotation='vertical')
    plt.title(r'Varying resolution. Polarization Y. wavelength $\lambda \approx 390 \mu m $, x-axis read as (height, width, source position)')
    plt.ylabel('LEE (%)')
    plt.plot(index,[x * 100 for x in ploty1],color='g',marker='.',ls='--',label='resolution 30')
    plt.plot(index,[x * 100 for x in ploty2],color='b',marker='^',ls='-.',label='resolution 60')
    plt.plot(index,[x * 100 for x in ploty3],color='m',marker='o',ls=':',label='resolution 120')
    plt.legend(loc='best')
    plt.show()
if av_time:
    db1r=sys.argv[1]
    db1 = read("db/initial_results/{}.json".format(db1r))
    time = db_to_array(db1,"result",'Elapsed time (min)')
    FF =  db_to_array(db1,"result",'flux_ratio') 
    FF_ABOVE = []
    FF_CURR = 0
    FF_ABOVEWL=[]
    FF_CURRWL=0
    for i in range(len(FF)):
        FF_CURR = FF[i][1]
        FF_CURR = FF_CURR[3]
        FF_ABOVE.append(FF_CURR)
        for k in range(len(FF[i][1])):
            FF_CURRWL=FF[i][1][k]
            FF_ABOVEWL.append(FF_CURRWL)
    Total_ff = 0
    Total_ffWL = 0
  #  print('len ',len(FF_ABOVEWL))
  #  print(FF_ABOVE)
  #  print(FF_ABOVEWL)
    for n in range(len(FF_ABOVE)):
        Total_ff += FF_ABOVE[n]
    for k in range(len(FF_ABOVEWL)):
        Total_ffWL += FF_ABOVEWL[k]
   #     print(Total_ffWL,FF_ABOVEWL[k])
    Total_Time = 0
    for n in range(len(time)):
        Total_Time += time[n]
    #print('len ff above, ff above wl',len(FF_ABOVE),len(FF_ABOVEWL),Total_ff,Total_ffWL)
    print('Total time (min):',Total_Time, 'Average time (min):', Total_Time/len(time)) 
    print('Average LEE 740 nm:',Total_ff/len(FF_ABOVE),'Average LEE over all wavlengths:',Total_ffWL/len(FF_ABOVEWL))
if time_processing:
    db1 = read("db/initial_results/2convtime1.json")   
    db2 = read("db/initial_results/2convtime3.json")
    db3 = read("db/initial_results/2convtime5.json")
    datax = db_to_array(db2,"pyramid","pyramid_height")
    datay1 = db_to_array(db1,"result","flux_ratio")
    datay2 = db_to_array(db2,"result","flux_ratio")
    datay3 = db_to_array(db3,"result","flux_ratio")
    ploty1=[]
    ploty2=[]
    ploty3=[]
    #data[n][v][k] n - pyr height v - 0/1 above or below [k] - freq
   # print('data raw',datay1)
  #  print(datay1[2][0][1])
    k=0 #freq index
    k=int(sys.argv[1])
    print(len(datay2))
    print(len(datay3))
    for n in range(len(datay2)):
        ploty1.append(datay1[n][0][k]) 
        ploty2.append(datay2[n][0][k])
        ploty3.append(datay3[n][0][k])
    print('ff30=',ploty1)
    print('ff60=',ploty2)
    print('ff120=',ploty3)
if dpml_processing:
    k=1 #frequency index 
    db1 = read("db/initial_results/{}.json".format(db1r))   
    db2 = read("db/initial_results/{}.json".format(db2r))
    db4 = read("db/initial_results/{}.json".format(db3r))
    X_Chosen30 = []
    X_Chosen60=[]
    X_Chosen120=[]
    for pyramid in db1:
        #print('pyramid height',pyramid["pyramid"]["pyramid_height"],'pyramid width',pyramid["pyramid"]["pyramid_width"])
        if pyramid["pyramid"]["pyramid_width"]==2:
            X_Chosen30.append(pyramid["result"]["flux_ratio"][k])
    for pyramid in db2:
        #print('pyramid height',pyramid["pyramid"]["pyramid_height"],'pyramid width',pyramid["pyramid"]["pyramid_width"])
        if pyramid["pyramid"]["pyramid_width"]==2:
            X_Chosen60.append(pyramid["result"]["flux_ratio"][k])
    for pyramid in db4:
        #print('pyramid height',pyramid["pyramid"]["pyramid_height"],'pyramid width',pyramid["pyramid"]["pyramid_width"])
        if pyramid["pyramid"]["pyramid_width"]==2:
            X_Chosen120.append(pyramid["result"]["flux_ratio"][k])

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
   # error=0
   # for n in range(len(ploty1)):
      #  error+=abs(ploty1[n]-ploty4[n])
    #print('error',error)
    print('ffpml1=',ploty1)
    print('ffpml2=',ploty2)
    print('ffpml4=',ploty4)
    print('ff30=',X_Chosen30)
    print('ff60=',X_Chosen60)
    print('ff120=',X_Chosen120)
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
