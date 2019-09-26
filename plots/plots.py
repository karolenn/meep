import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math			
"#######RESOLUTION PLOTS########################################"
"VARYING SOURCE POS"
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
#plt.plot(sp,ff240,color='r',marker='v',ls='dotted',label='resolution 240')
plt.title(r'$\lambda \approx 500$, pyramid height=1, pyramid width=1')
plt.legend(loc='best')
plt.xlabel('Source position')
plt.ylabel('Flux ratio')
plt.show()
ff30=[0.009388449, 0.005825919, 0.003659264, 0.004322376, 0.00837728, 0.013525549, 0.015928343, 0.016809371, 0.01726751, 0.014999034]
ff60=[0.005008834, 0.003198536, 0.002886876, 0.004161084, 0.006813861, 0.00953454, 0.01004405, 0.009736461, 0.010512841, 0.014039327]
ff120=[0.0049, 0.0036, 0.0031, 0.0036, 0.0053, 0.0074, 0.0083, 0.0087, 0.0105, 0.0162]
plt.plot(sp,ff30,color='g',marker='.',ls='--',label='resolution 30')
plt.plot(sp,ff60,color='b',marker='^',ls='-.',label='resolution 60')
plt.plot(sp,ff120,color='m',marker='o',ls=':',label='resolution 120')
#plt.plot(sp,ff240,color='r',marker='v',ls='dotted',label='resolution 240')
plt.title(r'$\lambda \approx 500$, pyramid height=1, pyramid width=1, Source polarization Z')
plt.legend(loc='best')
plt.xlabel('Source position')
plt.ylabel('Flux ratio')
plt.show()
"VARYING WIDTH"
##res30, 60, 120 freqs freqs [1.0, 1.2222222222222223, 1.4444444444444444, 1.6666666666666667, 1.8888888888888888, 2.111111111111111, 2.3333333333333335, 2.5555555555555554, 2.7777777777777777, 3.0]
#lambda = 690, 600, 530, 470, 430, 
#vi vill jämföra lambda 690, 530, 430. Alltså k=2,4,6
#skit i frekvens k=0,1,7,8,9
#k freq = 5 h=1 w=1-5 s=0.2
#w=[1,2,3,4,5]
#ff30=[0.066, 0.0597, 0.0569, 0.0534, 0.0434]
#ff60=[0.0832, 0.0438, 0.052, 0.0475, 0.0441]
#ff120=[0.0829, 0.0455, 0.0503, 0.0468, 0.0401]
#plt.plot(w,ff30,color='g',marker='.',ls='--',label='resolution 30')
#plt.plot(w,ff60,color='b',marker='^',ls='-.',label='resolution 60')
#plt.plot(w,ff120,color='m',marker='o',ls=':',label='resolution 120')
#plt.legend(loc='best')
#plt.xlabel('Pyramid width')
#plt.ylabel('Flux ratio')
#plt.show()
#k = 2 #height = 1 s=0.2
w=[1,2,3,4,5]
ff30= [0.0403, 0.0552, 0.04, 0.019, 0.0183]
ff60= [0.0492, 0.0523, 0.0338, 0.0268, 0.0221]
ff120= [0.0496, 0.0533, 0.0343, 0.0271, 0.0224]
plt.title(r'$\lambda \approx 700$, pyramid height=1, source position=0.2')
plt.plot(w,ff30,color='g',marker='.',ls='--',label='resolution 30')
plt.plot(w,ff60,color='b',marker='^',ls='-.',label='resolution 60')
plt.plot(w,ff120,color='m',marker='o',ls=':',label='resolution 120')
plt.legend(loc='best')
plt.xlabel('Pyramid width')
plt.ylabel('Flux ratio')
plt.show()
#k = 4 #height = 2
ff30= [0.1305, 0.1067, 0.1243, 0.093, 0.1001]
ff60= [0.1136, 0.0836, 0.1219, 0.0797, 0.0866]
ff120= [0.113, 0.0846, 0.1208, 0.0775, 0.0889]
plt.title(r'$\lambda \approx 500$, pyramid height=2, source position=0.2')
plt.plot(w,ff30,color='g',marker='.',ls='--',label='resolution 30')
plt.plot(w,ff60,color='b',marker='^',ls='-.',label='resolution 60')
plt.plot(w,ff120,color='m',marker='o',ls=':',label='resolution 120')
plt.legend(loc='best')
plt.xlabel('Pyramid width')
plt.ylabel('Flux ratio')
plt.show()
#k = 6 #height = 3
ff30= [0.3037, 0.0078, 0.0466, 0.0848, 0.117]
ff60= [0.1922, 0.0156, 0.0564, 0.0707, 0.1226]
ff120= [0.2018, 0.0196, 0.0635, 0.07, 0.1135]
plt.title(r'$\lambda \approx 400$, pyramid height=3, source position=0.2')
plt.plot(w,ff30,color='g',marker='.',ls='--',label='resolution 30')
plt.plot(w,ff60,color='b',marker='^',ls='-.',label='resolution 60')
plt.plot(w,ff120,color='m',marker='o',ls=':',label='resolution 120')
plt.legend(loc='best')
plt.xlabel('Pyramid width')
plt.ylabel('Flux ratio')
plt.show()
##############
"VARYING HEIGHT"
###############
#k freq =4  h=1-5 w=1 s=0.2
#
#ff30=[0.0609, 0.1305, 0.2451, 0.3865, 0.481]
#ff60=[0.0758, 0.1136, 0.1462, 0.1922, 0.3217] 
#ff120=[0.075, 0.113, 0.1431, 0.1894, 0.2391]
#plt.plot(h,ff30,color='g',marker='.',ls='--',label='resolution 30')
#plt.plot(h,ff60,color='b',marker='^',ls='-.',label='resolution 60')
#plt.plot(h,ff120,color='m',marker='o',ls=':',label='resolution 120')
#plt.legend(loc='best')
#plt.xlabel('Pyramid height')
#plt.ylabel('Flux ratio')
#plt.show()
h=[1,2,3,4,5]
#k=2
########WIDTH = 1 #############
"#error between last result 60,120 believed due to time not really decayed. FF was converged but total flux was not."
ff30= [0.0403, 0.0737, 0.0296, 0.0724, 0.2698]
ff60= [0.0492, 0.0849, 0.1068, 0.1262, 0.0333]
ff120= [0.0496, 0.0762, 0.1078, 0.1188, 0.1386]
plt.plot(h,ff30,color='g',marker='.',ls='--',label='resolution 30')
plt.plot(h,ff60,color='b',marker='^',ls='-.',label='resolution 60')
plt.plot(h,ff120,color='m',marker='o',ls=':',label='resolution 120')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 700$, pyramid width=1, source position=0.2')
plt.xlabel('Pyramid height')
plt.ylabel('Flux ratio')
plt.show()

#k=4
ff30= [0.0609, 0.1305, 0.2451, 0.3865, 0.481]
ff60= [0.0758, 0.1136, 0.1462, 0.1922, 0.3217]
ff120= [0.075, 0.113, 0.1431, 0.1894, 0.2391]
plt.plot(h,ff30,color='g',marker='.',ls='--',label='resolution 30')
plt.plot(h,ff60,color='b',marker='^',ls='-.',label='resolution 60')
plt.plot(h,ff120,color='m',marker='o',ls=':',label='resolution 120')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 500$, pyramid width=1, source position=0.2')
plt.xlabel('Pyramid height')
plt.ylabel('Flux ratio')
plt.show()

#k=6
ff30= [0.0846, 0.131, 0.3037, 0.3467, 0.2933]
ff60= [0.0734, 0.144, 0.1922, 0.233, 0.2714]
ff120= [0.0735, 0.137, 0.2018, 0.2281, 0.2216]
plt.plot(h,ff30,color='g',marker='.',ls='--',label='resolution 30')
plt.plot(h,ff60,color='b',marker='^',ls='-.',label='resolution 60')
plt.plot(h,ff120,color='m',marker='o',ls=':',label='resolution 120')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 400$, pyramid width=1, source position=0.2')
plt.xlabel('Pyramid height')
plt.ylabel('Flux ratio')
plt.show()

######WIDTH = 2#################
#k=2
"#error between last result 60,120 believed due to time not really decayed. FF was converged but total flux was not."
ff30= [0.0552, 0.0789, 0.0759, 0.1008, 0.1709]
ff60= [0.0523, 0.0726, 0.1171, 0.1071, 0.098]
ff120= [0.0533, 0.0687, 0.1194, 0.1044, 0.1004]
plt.plot(h,ff30,color='g',marker='.',ls='--',label='resolution 30')
plt.plot(h,ff60,color='b',marker='^',ls='-.',label='resolution 60')
plt.plot(h,ff120,color='m',marker='o',ls=':',label='resolution 120')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 700$, pyramid width=2, source position=0.2')
plt.xlabel('Pyramid height')
plt.ylabel('Flux ratio')
plt.show()

#k=4
ff30= [0.056, 0.1067, 0.0864, 0.0165, 0.1096]
ff60= [0.04, 0.0836, 0.0946, 0.0251, 0.0359]
ff120= [0.0405, 0.0846, 0.0983, 0.0283, 0.0338]
plt.plot(h,ff30,color='g',marker='.',ls='--',label='resolution 30')
plt.plot(h,ff60,color='b',marker='^',ls='-.',label='resolution 60')
plt.plot(h,ff120,color='m',marker='o',ls=':',label='resolution 120')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 500$, pyramid width=2, source position=0.2')
plt.xlabel('Pyramid height')
plt.ylabel('Flux ratio')
plt.show()

#k=6
ff30= [0.0829, 0.0564, 0.0078, 0.1468, 0.2915]
ff60= [0.0567, 0.142, 0.0156, 0.0861, 0.2201]
ff120= [0.0581, 0.1536, 0.0196, 0.0739, 0.2187]
plt.plot(h,ff30,color='g',marker='.',ls='--',label='resolution 30')
plt.plot(h,ff60,color='b',marker='^',ls='-.',label='resolution 60')
plt.plot(h,ff120,color='m',marker='o',ls=':',label='resolution 120')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 400$, pyramid width=2, source position=0.2')
plt.xlabel('Pyramid height')
plt.ylabel('Flux ratio')
plt.show()

"############TIME PLOTS###########################################"
#Errors in resolution calcs is due to total flux not really converged. 
###Sufficient plots to argue for resolution 60? Next argue for time convergence:
#freqs=1.4,2.2.6, lambda=715, 500, 385
#cutoff=2 this was run for source position 0.1 and results converge for 3e..?
#1e:8,10,12,14,18
#3e: 40,50,62,66,82
#5e: 172,198,279,251,327
h=[1,2,3,4,5]
#k=0 OLD ONES
ff1e= [0.058163, 0.104982, 0.101295, 0.059538, 0.106018]
ff3e= [0.073477, 0.089777, 0.106469, 0.104877, 0.105709]
ff5e= [0.064533, 0.091159, 0.107745, 0.108996, 0.106176304]
#NEVER CORRECTS ONES BELOW OUTCOMMENTED, SHOWS THAT RESULTS HAVENT CONVERGED IN TIME
#ff1e= [0.02137188, 0.000103422, 0.000151929, 0.002378566, 0.033845666, 0.008802663, 0.005158022, 0.018947004]
#ff3e= [0.04691264, 0.001203261, 0.000497963, 0.000204898, 0.033845666, 0.046520769, 0.025727793, 0.049638922]
#ff5e= [0.04691264, 0.059247406, 0.000497963, 0.000204898, 0.000253441, 0.046520769, 0.079518517, 0.049638922]
ff1e[:] = [x * 100 for x in ff1e]
ff3e[:] = [x * 100 for x in ff3e]
ff5e[:] = [x * 100 for x in ff5e]
plt.plot(h,ff1e,color='y',marker='.',ls='--',label='Probe sensitivity 1e-1')
plt.plot(h,ff3e,color='k',marker='^',ls='-.',label='Probe sensitivity 1e-3')
plt.plot(h,ff5e,color='c',marker='o',ls=':',label='Probe sensitivity 1e-5')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 700$, pyramid width=1, source position=0.2, Polarization Y')
plt.xlabel('Pyramid width')
plt.ylabel('LEE (%)')
plt.savefig('convtime700.pdf') 
plt.show()
#k=1
ff1e= [0.049918, 0.083539, 0.102278, 0.109035, 0.119667]
ff3e= [0.050036, 0.083464, 0.102764, 0.10921, 0.119795]
ff5e= [0.050029, 0.083437, 0.102766, 0.109219, 0.119795968]
#ff1e= [0.07416225, 0.208664777, 0.18257614, 0.181509061, 0.196683345, 0.190870574, 0.41676869, 0.532988878]
#ff3e= [0.074157953, 0.098179511, 0.084058733, 0.134025995, 0.196683345, 0.187593356, 0.134565832, 0.129501948]
#ff5e= [0.074157953, 0.090472877, 0.084058733, 0.134025995, 0.183556356, 0.187593356, 0.134364168, 0.129501948]
ff1e[:] = [x * 100 for x in ff1e]
ff3e[:] = [x * 100 for x in ff3e]
ff5e[:] = [x * 100 for x in ff5e]
plt.plot(h,ff1e,color='y',marker='.',ls='--',label='Probe sensitivity 1e-1')
plt.plot(h,ff3e,color='k',marker='^',ls='-.',label='Probe sensitivity 1e-3')
plt.plot(h,ff5e,color='c',marker='o',ls=':',label='Probe sensitivity 1e-5')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 500$, pyramid width=1, source position=0.2, Polarization Y')
plt.xlabel('Pyramid width')
plt.ylabel('LEE (%)')
plt.savefig('convtime500.pdf') 
plt.show()
#k=2
ff1e= [0.047181, 0.086435, 0.114375, 0.187308, 0.150384]
ff3e= [0.048941, 0.07694, 0.103934, 0.12796, 0.140555]
ff5e= [0.047511, 0.077334, 0.103717, 0.126637, 0.140422462]
#ff1e= [0.066228711, 0.001351743, 0.000769494, 0.007414897, 0.093136273, 0.03461264, 0.038136756, 0.153459475]
#ff3e= [0.061926882, 0.009460906, 0.001884746, 0.001659863, 0.093136273, 0.104701684, 0.072603565, 0.102389021]
#ff5e= [0.061926882, 0.083803753, 0.001884746, 0.001659863, 0.001799979, 0.104701684, 0.114912052, 0.102389021]
ff1e[:] = [x * 100 for x in ff1e]
ff3e[:] = [x * 100 for x in ff3e]
ff5e[:] = [x * 100 for x in ff5e]
plt.plot(h,ff1e,color='y',marker='.',ls='--',label='Probe sensitivity 1e-1')
plt.plot(h,ff3e,color='k',marker='^',ls='-.',label='Probe sensitivity 1e-3')
plt.plot(h,ff5e,color='c',marker='o',ls=':',label='Probe sensitivity 1e-5')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 400$, pyramid width=1, source position=0.2, Polarization Y')
plt.xlabel('Pyramid width')
plt.ylabel('LEE (%)')
plt.savefig('convtime400.pdf') 
plt.show()
"###################DPML PLOTS######################"
#vi vill jämföra lambda 690, 530, 430. Alltså k=2,4,6
#k=2
h=[1.0, 1.4444444444444444, 1.8888888888888888, 2.333333333333333, 2.7777777777777777, 3.2222222222222223, 3.6666666666666665, 4.111111111111111, 4.555555555555555, 5.0]
ff1= [0.0492, 0.0631, 0.0748, 0.0962, 0.1078, 0.0281, 0.1165, 0.2321, 0.2456, 0.0333]
ff3= [0.0492, 0.0631, 0.0748, 0.0962, 0.1078, 0.0281, 0.1165, 0.2321, 0.2456, 0.0333]
ff5= [0.0492, 0.0631, 0.0748, 0.0962, 0.1078, 0.0281, 0.1165, 0.2321, 0.2456, 0.0333]
ff1[:] = [x * 100 for x in ff1]
ff3[:] = [x * 100 for x in ff3]
ff5[:] = [x * 100 for x in ff5]
plt.plot(h,ff1,color='indigo',marker='.',ls='--',label='PML depth 0.1')
plt.plot(h,ff3,color='navy',marker='^',ls='-.',label='PML depth 0.2')
plt.plot(h,ff5,color='hotpink',marker='o',ls=':',label='PML depth 0.4')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 700$, pyramid width=1, source position=0.2, Polarization Y')
plt.xlabel('Pyramid height')
plt.ylabel('LEE (%)')
plt.savefig('convpml700.pdf') 
plt.show()
#k=4
ffpml1= [0.0758, 0.0982, 0.1105, 0.1208, 0.1353, 0.2195, 0.1774, 0.3517, 0.4039, 0.3217]
ffpml2= [0.0758, 0.0982, 0.1105, 0.1209, 0.1353, 0.2195, 0.1774, 0.3517, 0.4039, 0.3209]
ffpml4= [0.0756, 0.0982, 0.1105, 0.1209, 0.1353, 0.2195, 0.1774, 0.3517, 0.4039, 0.3209]
ffpml1[:] = [x * 100 for x in ffpml1]
ffpml2[:] = [x * 100 for x in ffpml2]
ffpml4[:] = [x * 100 for x in ffpml4]
plt.plot(h,ffpml1,color='indigo',marker='.',ls='--',label='PML depth 0.1')
plt.plot(h,ffpml2,color='navy',marker='^',ls='-.',label='PML depth 0.2')
plt.plot(h,ffpml4,color='hotpink',marker='o',ls=':',label='PML depth 0.4')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 500$, pyramid width=1, source position=0.2, Polarization Y')
plt.xlabel('Pyramid height')
plt.ylabel('LEE (%)')
plt.savefig('convpml500.pdf') 
plt.show()
#k=6
ffpml1= [0.0734, 0.089, 0.1225, 0.1785, 0.2006, 0.271, 0.2295, 0.3752, 0.3629, 0.2714]
ffpml2= [0.0733, 0.0889, 0.1225, 0.1782, 0.2007, 0.271, 0.2293, 0.3752, 0.3629, 0.2716]
ffpml4= [0.0733, 0.0889, 0.1225, 0.1782, 0.2006, 0.271, 0.2293, 0.3752, 0.3627, 0.2716]
ffpml1[:] = [x * 100 for x in ffpml1]
ffpml2[:] = [x * 100 for x in ffpml2]
ffpml4[:] = [x * 100 for x in ffpml4]
plt.plot(h,ffpml1,color='indigo',marker='.',ls='--',label='PML depth 0.1')
plt.plot(h,ffpml2,color='navy',marker='^',ls='-.',label='PML depth 0.2')
plt.plot(h,ffpml4,color='hotpink',marker='o',ls=':',label='PML depth 0.4')
plt.legend(loc='best')
plt.title(r'$\lambda \approx 400$, pyramid width=1, source position=0.2, Polarization Y')
plt.xlabel('Pyramid height')
plt.ylabel('LEE (%)')
plt.savefig('convpml400.pdf') 
plt.show()


