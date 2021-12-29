from functionality.api import *
from functionality.functions import *
import matplotlib.pyplot as plt

sim_name = "DE_first_result"

def plot_LE(sim_name):
    db = read("../db/initial_results/{}.json".format(sim_name))

    len_db = len(db)

    all_results = [0]*len_db
    total_flux_results = [0]*len_db
    source_flux_results = [0]*len_db
    source_pos = [0]*len_db



    total_flux_GaN = [0.00011428871, 0.0250669074, 0.8222765772, 5.30342497776, 6.72770614219, 1.68183821152, 0.08313836489, 0.00077824323]
    s_flux_GaN = [0.0001591884196414295, 0.025045174820939542, 0.8212780288510312, 5.293335569705324, 6.714833870606542, 1.6796636762639658, 0.08292949836437485, 0.0008176334840038225]
    LDOS_GaN = [8.573990952416164, 7.996704390999339, 8.983391563270619, 10.180754634708808, 11.487315627492945, 12.895229874597154, 14.45193173631105, 16.494400901543397]
    freqs = [1.4700000000000002, 1.5785714285714287, 1.6871428571428573, 1.7957142857142858, 1.9042857142857144, 2.012857142857143, 2.1214285714285714, 2.23]

    lambda_wl = []
    for n in range(len(freqs)):
        lambda_wl.append(round(1000/freqs[n],1)) 

    nfreq = db[0]["pyramid"]["number_of_freqs"]

    RE = [0]*len_db
    RE = [RE]*nfreq

    Purcell = [0]*len_db
    Purcell = [Purcell]*nfreq

    for i in range(len_db):
        source_pos[i] = db[i]["pyramid"]["source_position"][2]*1000

    #calculate RE
    for k in range(nfreq):
        tmp = [0]*len_db
        tmp_purcell = 0
        for i in range(len_db):
            source_flux = db[i]["result"]["source_flux"][k]
            total_flux = db[i]["result"]["total_flux"][k]
            print('s,t', source_flux, total_flux, total_flux / source_flux)
            tmp[i] = 100 - 100*total_flux / source_flux

        RE[k] = tmp

    print(' ')
    #calculate Purcell
    for k in range(nfreq):
        tmp = [0]*len_db
        for i in range(len_db):
            source_flux = db[i]["result"]["source_flux"][k]
            flux_GaN = total_flux_GaN[k]
            print('s,g', source_flux, flux_GaN, source_flux / flux_GaN)
            tmp[i] = 100 * source_flux / flux_GaN

        Purcell[k] = tmp

    colors = ['darkred','red', 'darkorange','limegreen','aquamarine','teal','navy', 'blue']
    for n in range(len(RE)):
        print('plot sp', source_pos)
        print('RE', RE[n])
        plt.title('1 micrometer wide base pyramid. 1 - total flux / source flux')
        plt.plot(source_pos, RE[n], marker='o', ls='--', label=str(lambda_wl[n])+"nm",color=colors[n])
        plt.ylabel('Absorption (%)')
        plt.xlabel('source position (nm)')
        plt.grid(visible=True)
        plt.legend(loc='best')

    plt.show()
    plt.savefig('Absorption.png')
    plt.clf()

    for n in range(len(Purcell)):
        plt.title('1 micrometer wide base pyramid. source flux (pyramid) / source flux (GaN)')
        plt.plot(source_pos, Purcell[n], marker='o', ls='--', label=str(lambda_wl[n])+"nm",color=colors[n])
        plt.ylabel('Purcell factor (%)')
        plt.xlabel('source position (nm)')
        plt.grid(visible=True)
        plt.legend(loc='best')

    plt.savefig('Purcell.png')
    #print(total_flux_results)
    #print(source_flux_results)
    print(source_pos)

    extraction_eff = [0]*len(Purcell)

    for n in range(len(Purcell)):
        plt.title('1 micrometer wide base pyramid. source flux (pyramid) / source flux (GaN)')
        plt.plot(source_pos, Purcell[n]*RE[n], marker='o', ls='--', label=str(lambda_wl[n])+"nm",color=colors[n])
        plt.ylabel('Super (%)')
        plt.xlabel('source position (nm)')
        plt.grid(visible=True)
        plt.legend(loc='best')

    plt.savefig('Purcell.png')

plot_LE(sim_name)