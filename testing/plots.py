import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math			

#res30 varying source position h=w=1 freq=2 (?) cutoff=2
sp= [0.01, 0.05333333333333334, 0.09666666666666666, 0.14, 0.18333333333333335, 0.22666666666666668, 0.27, 0.31333333333333335, 0.3566666666666667, 0.4]
ff30=[0.126, 0.1039, 0.0467, 0.0348, 0.0878, 0.0935, 0.0493, 0.0956, 0.1026, 0.0851]
#res60
ff60=[0.097, 0.0796, 0.0508, 0.0487, 0.0742, 0.0738, 0.065, 0.0867, 0.0858, 0.095]
#res120
ff120=[0.0985, 0.0806, 0.051, 0.0478, 0.0724, 0.0754, 0.0638, 0.0838, 0.0872, 0.0938]
#res240
ff240=[0.0994, 0.0813, 0.0511, 0.0475, 0.0714, 0.0748, 0.0633, 0.0833, 0.0873, 0.093]
plt.plot(sp,ff30,color='g',marker='.',ls='--',label='resolution 30')
plt.plot(sp,ff60,color='b',marker='^',ls='-.',label='resolution 60')
plt.plot(sp,ff120,color='m',marker='o',ls=':',label='resolution 120')
plt.plot(sp,ff240,color='r',marker='v',ls='dotted',label='resolution 240')
plt.legend(loc='best')
plt.xlabel('Source position')
plt.ylabel('Flux ratio')
plt.show()
##res30, 60, 120 freqs [1.5, 1.6111111111111112, 1.7222222222222223, 1.8333333333333335, 1.9444444444444444, 2.0555555555555554, 2.166666666666667, 2.2777777777777777, 2.388888888888889, 2.5]
#k freq = 5 h=1 w=1-5 s=0.2
ff30=[0.066, 0.0597, 0.0569, 0.0534, 0.0434]
ff60=[0.0832, 0.0438, 0.052, 0.0475, 0.0441]
ff120=[0.0829, 0.0455, 0.0503, 0.0468, 0.0401]
#k freq =4  h=1-5 w=1 s=0.2
ff30=[0.0609, 0.1305, 0.2451, 0.3865, 0.481]
ff60=[0.0758, 0.1136, 0.1462, 0.1922, 0.3217] #error between last result 60,120 believed due to time not really decayed. FF was converged but total flux was not.
ff120=[0.075, 0.113, 0.1431, 0.1894, 0.2391]
###Sufficient plots to argue for resolution 60? Next argue for time convergence:
#cutoff=2 this was run for source position 0.1 and results converge for 3e..?
#1e:8,10,12,14,18
#3e: 40,50,62,66,82
#5e: 172,198,279,251,327


