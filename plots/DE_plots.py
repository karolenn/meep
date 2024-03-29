from functionality.api import *
from functionality.functions import *
from functionality.QWrapper_functions import unpack_3_list
import matplotlib.pyplot as plt
import math

from pathlib import Path
#sim_name = "DE_xpol_CL_Pd"
#sim_name = "DE_first_result_3"

freqs = [1.4700000000000002, 1.5785714285714287, 1.6871428571428573, 1.7957142857142858, 1.9042857142857144, 2.012857142857143, 2.1214285714285714, 2.23]

lambda_wl = []
for n in range(len(freqs)):
    lambda_wl.append(round(1000/freqs[n],1)) 
colors = ['darkred','red', 'darkorange','limegreen','aquamarine','teal','navy', 'blue']
def plot_LE(sim_name):
    db = read("../db/initial_results/{}.json".format(sim_name))

    len_db = len(db)

    all_results = [0]*len_db
    total_flux_results = [0]*len_db
    source_flux_results = [0]*len_db
    source_pos = [0]*len_db
    freqs = [1.4700000000000002, 1.5785714285714287, 1.6871428571428573, 1.7957142857142858, 1.9042857142857144, 2.012857142857143, 2.1214285714285714, 2.23]
    lambda_wl = []
    for n in range(len(freqs)):
        lambda_wl.append(round(1000/freqs[n],1)) 
    colors = ['darkred','red', 'darkorange','limegreen','aquamarine','teal','navy', 'blue']

    total_flux_GaN = [0.00011428871, 0.0250669074, 0.8222765772, 5.30342497776, 6.72770614219, 1.68183821152, 0.08313836489, 0.00077824323]
    s_flux_GaN = [0.0001591884196414295, 0.025045174820939542, 0.8212780288510312, 5.293335569705324, 6.714833870606542, 1.6796636762639658, 0.08292949836437485, 0.0008176334840038225]
    LDOS_GaN = [8.573990952416164, 7.996704390999339, 8.983391563270619, 10.180754634708808, 11.487315627492945, 12.895229874597154, 14.45193173631105, 16.494400901543397]

    total_flux_GaN_big = [7.09345e-05, 0.02491539364, 0.81795362469, 5.27149694554, 6.68203329739, 1.66905158049, 0.08237088817, 0.00074259653]
    s_flux_GaN_big = [0.00013138826970230657, 0.02498465339208524, 0.8158905262374332, 5.252010400265657, 6.665232389201147, 1.6627468348347554, 0.08240902192610353, 0.0007895323656438263]
    LDOS_GaN_big = [7.266880424385067, 7.854357146929161, 8.9417050360065, 10.15248309213125, 11.45512090423561, 12.852138409663931, 14.359943808947058, 16.130830123408547]

    freqs = [1.4700000000000002, 1.5785714285714287, 1.6871428571428573, 1.7957142857142858, 1.9042857142857144, 2.012857142857143, 2.1214285714285714, 2.23]

    lambda_wl = []
    for n in range(len(freqs)):
        lambda_wl.append(round(1000/freqs[n],1)) 

    nfreq = db[0]["pyramid"]["number_of_freqs"]

    pyramid_width = db[0]["pyramid"]["pyramid_width"]

    RE = [0]*len_db
    RE = [RE]*nfreq

    Purcell = [0]*len_db
    Purcell = [Purcell]*nfreq

    for i in range(len_db):
        source_pos[i] = db[i]["pyramid"]["source_position"][2]*1000

    #calculate Absorption
    for k in range(nfreq):
        tmp = [0]*len_db
        tmp_purcell = 0
        for i in range(len_db):
            source_flux = db[i]["result"]["source_flux"][k]
            total_flux = db[i]["result"]["total_flux"][k]
            #print('s,t', source_flux, total_flux, 100 - 100*total_flux / source_flux)
            tmp[i] = 100 - 100*total_flux / source_flux

        RE[k] = tmp

    #calculate Purcell
    for k in range(nfreq):
        tmp = [0]*len_db
        for i in range(len_db):
            source_flux = db[i]["result"]["source_flux"][k]
            flux_GaN = total_flux_GaN[k]
            #print('s,g', source_flux, flux_GaN, source_flux / flux_GaN)
            tmp[i] = 100 * source_flux / flux_GaN

        Purcell[k] = tmp

    #calculate light extraction ff
    LE = [0]*len_db
    LE = [LE]*nfreq
    for k in range(nfreq):
        tmp = [0]*len_db
        ff_angle = 0
        for i in range(len_db):
            source_flux = db[i]["result"]["source_flux"][k]
            try:
                ff_angle = db[i]["result"]["ff_at_angle"][k]
            except:
                ff_angle = 0
            #print('s,t', source_flux, ff_angle, 100*ff_angle / source_flux)
            tmp[i] = 100*ff_angle / source_flux

        LE[k] = tmp
    #calculate light below

    #calculate light extraction bottom
    LE_bottom = [0]*len_db
    LE_bottom = [LE_bottom]*nfreq
    for k in range(nfreq):
        tmp = [0]*len_db
        for i in range(len_db):
            source_flux = db[i]["result"]["source_flux"][k]
            flux_bottom = db[i]["result"]["flux_bottom"][k]
            tmp[i] = 100*flux_bottom / source_flux

        LE_bottom[k] = tmp

    colors = ['darkred','red', 'darkorange','limegreen','aquamarine','teal','navy', 'blue']
    for n in range(len(RE)):
        plt.title('{} μm wide pyramid. 1 - total flux / source flux.'.format(pyramid_width))
        plt.plot(source_pos, RE[n], marker='o', ls='--', label=str(lambda_wl[n])+"nm",color=colors[n])
        plt.ylabel('Absorption (%)')
        plt.xlabel('source position (nm)')
        plt.grid(visible=True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()


    Path(sim_name).mkdir(parents=True, exist_ok=True)

    plt.show()
    plt.savefig(sim_name+'/'+'Absorption_{}.png'.format(sim_name), dpi=100)
    plt.clf()

    for n in range(len(LE)):
        #print('plot sp', source_pos)
        #print('LE', LE[n])
        plt.title('{} μm wide pyramid. flux far_field / source flux.'.format(pyramid_width))
        plt.plot(source_pos, LE[n], marker='o', ls='--', label=str(lambda_wl[n])+"nm",color=colors[n])
        plt.ylabel('far field LE (%)')
        plt.xlabel('source position (nm)')
        plt.grid(visible=True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

    plt.show()
    plt.savefig(sim_name+'/'+'LE_{}.png'.format(sim_name))
    plt.clf()

    for n in range(len(LE_bottom)):
        #print('plot sp', source_pos)
        #print('LE bottom of simulation', LE[n])
        plt.title('{} μm wide pyramid. flux bottom / source flux.'.format(pyramid_width))
        plt.plot(source_pos, LE_bottom[n], marker='o', ls='--', label=str(lambda_wl[n])+"nm",color=colors[n])
        plt.ylabel('bottom LE (%)')
        plt.xlabel('source position (nm)')
        plt.grid(visible=True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

    plt.show()
    plt.savefig(sim_name+'/'+'LE_bottom_{}.png'.format(sim_name))
    plt.clf()

    for n in range(len(Purcell)):
        plt.title('{} μm wide pyramid. source flux (pyramid) / source flux (GaN).'.format(pyramid_width))
        plt.plot(source_pos, Purcell[n], marker='o', ls='--', label=str(lambda_wl[n])+"nm",color=colors[n])
        plt.ylabel('Purcell factor (%)')
        plt.xlabel('source position (nm)')
        plt.grid(visible=True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

    plt.savefig(sim_name+'/'+'Purcell_{}.png'.format(sim_name))
    #print(total_flux_results)
    #print(source_flux_results)

def plot_far_field(db_name):
    #db_name = "test_simulate"
    #db_name = "DE_xpol_ff"
    #path = "../db"
    path = "../db/initial_results"
    print("{}/{}.json".format(path,db_name))
    #path = "../db/initial_results/{}.json".format(db_name)

    lambda_wl = []
    for n in range(len(freqs)):
        lambda_wl.append(round(1000/freqs[n],1)) 
    colors = ['darkred','red', 'darkorange','limegreen','aquamarine','teal','navy', 'blue']

    db = read("{}/{}.json".format(path,db_name))
    pyr = 1
    ff_at_angle = db[pyr]["result"]["ff_at_angle"]
    far_field, npts, nfreq, ff_angle, ph, fcen = extract_data_from_db(db_name,path, pyr)
    ff_values = return_field_values_from_ff(far_field)
    ff_pos = return_position_values_from_ff(far_field)

    Pr_array = calculate_poynting_values(ff_values)
    radius = 2*math.pow(ph,2)*fcen*2*10
    x,y,z = unpack_3_list(ff_pos)
    for k in range(nfreq):
        freq = k
        ff_at_angle_freq = ff_at_angle[freq]
        Pr_array_freq = get_poynting_per_frequency(Pr_array, freq)
        flux_per_freq = get_flux(Pr_array_freq,math.pi/ff_angle,npts,radius)
        theta_pts = int(math.sqrt((npts-1)/2))
        theta = math.pi / ff_angle
        theta_angles = np.linspace(0+theta/theta_pts,theta,theta_pts)
        cum_flux_per_angle = get_cum_flux_per_angle(Pr_array_freq, 3, npts, radius)
        cum_flux_per_angle_norm = [100*i / ff_at_angle_freq for i in cum_flux_per_angle]
        theta_angles_reversed = [i * -1 for i in theta_angles]
        flux_conc = cum_flux_per_angle + cum_flux_per_angle[::-1]
        theta_angles_conc = list(theta_angles)[::-1] + list(theta_angles_reversed)
        theta_angles_degrees = [i * 180/math.pi for i in theta_angles]
        plt.plot(theta_angles_degrees,cum_flux_per_angle_norm, label=str(lambda_wl[k])+"nm",color=colors[k])
        plt.legend(loc='best')
        plt.title('Share of far field flux within angle')
        plt.ylabel('(%) of far-field within angle')
        plt.xlabel('degrees')
        plt.savefig('angles.png')
        plt.grid(visible=True)

    plt.clf()

    norm_val = max(Pr_array_freq)
    Pr_array_freq_normed = [i / norm_val for i in Pr_array_freq]

    elements = int(1*npts/nfreq)

    theta_val = []
    phi_val = []
    for n in range(len(x)):
        theta_val.append(math.acos(z[n]/radius))
        if x[n] > 0:
            phi_val.append(y[n]/x[n])
        elif x[n] < 0 and y[n] >= 0:
            phi_val.append(y[n]/x[n]+math.pi)
        elif x[n] < 0 and y[n] < 0:
            phi_val.append(y[n]/x[n] - math.pi)
        elif x[n] == 0 and y[n] > 0:
            phi_val.append(math.pi/2)
        elif x[n] == 0 and y[n] < 0:
            phi_val.append(-math.pi/2)
        else: 
            phi_val.append(0)


    X,Y = np.meshgrid(theta_val,phi_val)
    Pr_mesh = np.meshgrid(Pr_array_freq_normed,Pr_array_freq_normed)
    #plt.Circle((0,0),15, fill=False)
    #plt.plot()
    plt.hexbin(x,y,Pr_array_freq_normed)
    plt.title('Sampled flux density')
    plt.xlabel('degrees')
    plt.ylabel('degrees')
    plt.colorbar(plt.hexbin(x,y,Pr_array_freq_normed))
    plt.savefig('flux2.png')


#plot_LE(sim_name)

#plot_far_field()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 :
        print('args: ',sys.argv)
        if sys.argv[1] == "plot_LE":
            plot_LE(sys.argv[2])
        elif sys.argv[1] == "plot_far_field":
            plot_far_field(sys.argv[2])
        else:
            print('Wrong input argument. First argument after DE_plots.py specifies which functions (plot_LE/plot_far_field) to use.')

    else:
        print("Specify function to run and db to use. Usage 'python DE_plots.py plot_LE DE_xpol_small' ")
        exit(0)