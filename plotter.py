from utils.api import *
from utils.functions import *
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from math import tan, pi

#freqs [1.0, 1.2222222222222223, 1.4444444444444444, 1.6666666666666667, 1.8888888888888888, 2.111111111111111, 2.3333333333333335, 2.5555555555555554, 2.7777777777777777, 3.0]


def RBF_plotter(sim_name, pts, ff_calc, args):
    db = read("db/initial_results/{}.json".format(sim_name))
    data = []
    print(args)
    for n in args:  # n is index in db
        data.append(db_to_array(db, "pyramid", n))
    ###This should be turned into a function. Same lines used in functions, merit function as well
    if len(data) == 2:
        pts = int(pts)
        db = read("db/initial_results/{}.json".format(sim_name))
        data = []
        print(args)
        for n in args:  # n is index in db
            data.append(db_to_array(db, "pyramid", n))
        sim_results = db_to_array(db, "result", "flux_ratio")
        #Withdraw ff_ratio above or under from results.
        # print('sim results passed to process',sim_results)
        # print('data recieved',data)
        if len(sim_results[0]) != 1:
            results = process_results(sim_results, ff_calc)

        datanx = data[0][:pts]
        datany = data[1][:pts]
        resultsn = results[:pts]
        # print('resultsn',resultsn)
        # print('datax',datanx)
        # print('datay',datany)
        #RETURN 5 best pts

        merge = list(zip(datanx, datany))
        count(merge)
        D1 = []
        D2 = []
        R = []
        print(len(datanx))
        for n in range(1, pts, 2):
            D1.append(datanx[n])
            D2.append(datany[n])
            R.append(results[n])
        #   datanx=D1
        #  datany=D2
        # resultsn=R
        # print(datanx)
        "PLOT SURFACE"
        minx = min(datanx)
        maxx = max(datanx)
        miny = min(datany)
        maxy = max(datany)
        minf = min(resultsn)
        maxf = max(resultsn)
        minx = 1
        maxx = 8
        miny = 0.01
        maxy = 0.9
        print(datanx)
        print(datany)
        x = np.linspace(minx, maxx, num=30)
        y = np.linspace(miny, maxy, num=30)
        f = np.linspace(minf, maxf, num=30)
        RBF_Func = Rbf(datanx, datany, resultsn, function='inverse')
        exploraion_pt = rand_pt_minmax([datanx, datany])
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(x, y)
        z = RBF_Func(X, Y)
        offset = 0
        ax.scatter(datanx, datany, [
                   100*i+1 for i in resultsn], s=3, alpha=1, c='k')
        #ax.scatter(exploraion_pt[0],exploraion_pt[1],c='r',marker='D',s=30)
        # ax.scatter(datanx,datany,[i * 100 for i in resultsn],c='k')
        ax.plot_surface(X, Y, 100*z, cmap=cm.jet, alpha=1)
        #   cset = ax.contourf(X, Y, z, zdir='z', offset=offset, cmap=cm.jet)
        #for i in range(len(datanx)):
        # ax.text(datanx[i],datany[i],resultsn[i],str(i))
        ax.set_xlabel('pyramid width $(\mu m)$')
        ax.set_ylabel('source position')
        ax.set_zlabel('LEE (%)')
        ax.view_init(elev=45, azim=135)
        #   ax.set_zlim(0,30)
        #   plt.tight_layout()
        plt.show()
        "PLOT HEATMAP"

        curr_max = 0
        index = 0
        n_best_pts = 5
        best_guys = []
        for k in range(n_best_pts):
            for i in range(len(resultsn)):
                if resultsn[i] > curr_max:
                    curr_max = resultsn[i]
                    index = i
            print(resultsn)
            best_guys.append((resultsn[index], 'height:', datanx[index]*tan(
                (pi*62)/180)/2, 'width:', datanx[index], 'sp:', datany[index]))
            curr_max = 0
            del resultsn[index]
            del datanx[index]
            del datany[index]
        print('best:', best_guys)
    elif len(data) == 3:
            pts = int(pts)
            db = read("db/initial_results/{}.json".format(sim_name))
            data = []
            print(args)
            for n in args:  # n is index in db
                data.append(db_to_array(db, "pyramid", n))
            sim_results = db_to_array(db, "result", "flux_ratio")
            #Withdraw ff_ratio above or under from results.
            # print('sim results passed to process',sim_results)
            # print('data recieved',data)
            print('len sim res',sim_results[0])
            print('sim res',sim_results)
            if len(sim_results[0]) != 1:
                results = process_results(sim_results, ff_calc)

            datanx = data[0][:pts]
            datany = data[1][:pts]
            datanz = data[2][:pts]
            resultsn = results[:pts]
            # print('resultsn',resultsn)
            # print('datax',datanx)
            # print('datay',datany)
            #RETURN 5 best pts

            merge = list(zip(datanx, datany, datanz))
            count(merge)
            D1 = []
            D2 = []
            R = []
            print(len(datanx))
            for n in range(1, pts, 2):
                D1.append(datanx[n])
                D2.append(datany[n])
                R.append(results[n])
            #   datanx=D1
            #  datany=D2
            # resultsn=R
            # print(datanx)
            "PLOT SURFACE"
            minx = min(datanx)
            maxx = max(datanx)
            miny = min(datany)
            maxy = max(datany)
            minz = min(datanz)
            maxz = max(datanx)
            minf = min(resultsn)
            maxf = max(resultsn)
            #minx=1
            #maxx=8
            #miny=0.01
            #maxy=0.9
            print(datanx)
            print(datany)
            x = np.linspace(minx, maxx, num=30)
            y = np.linspace(miny, maxy, num=30)
            z = np.linspace(minz, maxz, num=30)
            f = np.linspace(minf, maxf, num=30)
            RBF_Func = Rbf(datanx, datany, datanz,
                           resultsn, function='inverse')
            exploraion_pt = rand_pt_minmax([datanx, datany, datanz])
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            X, Y, Z = np.meshgrid(x, y, z)
            z = RBF_Func(X, Y, Z)
            offset = 0
            ax.scatter(datanx, datany, datanz, c=resultsn, s=12, cmap=cm.jet)
           # fig.colorbar(ax)
            #ax.scatter(exploraion_pt[0],exploraion_pt[1],c='r',marker='D',s=30)
            # ax.scatter(datanx,datany,[i * 100 for i in resultsn],c='k')
           # ax.plot_surface(X,Y,100*z,cmap=cm.jet,alpha=1)
            #   cset = ax.contourf(X, Y, z, zdir='z', offset=offset, cmap=cm.jet)
            #for i in range(len(datanx)):
            # ax.text(datanx[i],datany[i],resultsn[i],str(i))
            ax.set_xlabel('pyramid height $(\mu m)$')
            ax.set_ylabel('source position')
            ax.set_zlabel('pyramid height $(\mu m)$')
            ax.view_init(elev=45, azim=135)
            #   ax.set_zlim(0,30)
            #   plt.tight_layout()
            plt.show()
            "PLOT HEATMAP"

            curr_max = 0
            index = 0
            n_best_pts = 5
            best_guys = []
            for k in range(n_best_pts):
                for i in range(len(resultsn)):
                    if resultsn[i] > curr_max:
                        curr_max = resultsn[i]
                        index = i
                print(resultsn)
                best_guys.append((resultsn[index], 'height:', datanx[index]*tan(
                    (pi*62)/180)/2, 'width:', datanx[index], 'sp:', datany[index]))
                curr_max = 0
                del resultsn[index]
                del datanx[index]
                del datany[index]

            print('best:', best_guys)


def plotter(db, x, y):
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
    if (sys.argv[1]) == "plotter":
        plotter(sys.argv[2],sys.argv[3],sys.argv[4])
    elif (sys.argv[1]) == "RBF":
        print(sys.argv)
        print(sys.argv[3])
        RBF_plotter(sys.argv[2],sys.argv[3],sys.argv[4],[arg.strip() for arg in sys.argv[5:]])
    else:
        print("Not enough arguments")
        exit(0)
