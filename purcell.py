from functionality.api import *
from plots.plotter import RBF_plotter
import matplotlib.pyplot as plt

#([4.5625e-05, 0.049337749, 2.065924866, 3.463643881, 0.236057275, 0.000653973], None, None)
[4.5623e-05, 0.049343117, 2.066110881, 3.463624396, 0.236053262, 0.000654042]
Purcell_Bulk = [4.5623e-05, 0.049343117, 2.066110881, 3.463624396, 0.236053262, 0.000654042]
sim_name = "fybestabove3"
ff_calc = "Above"
db = read("db/initial_results/{}.json".format(sim_name))

values = []

#withdraw the flux_ratio results from initial_runs simulations"
RBF_plotter(sim_name, 45, False, ["pyramid_width","source_position"],Purcell_Bulk)