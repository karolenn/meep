import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math
import cmath			

x=np.linspace(0,10,1000)
Imx = []
for n in range(len(x)):
    if x[n] < 5:
        Imx.append(0)
    else:
        Imx.append(x[n]-5)
#print(Imx)
expi = []
for n in range(len(x)):
    #expi.append(math.exp(1j*(x[n]+1j*Imx[n])))
    expi.append(5*math.cos(x[n]*(2*math.pi))*math.exp(-Imx[n]))
#print(expi)
plt.plot(x,Imx,label='Im x')
plt.plot(x,expi,label='exp(ix)')
plt.plot([5,5,5],[-5,0,5],ls='--',color='k')
plt.text(6,4,'PML region')
plt.legend(loc='best')
plt.xlabel('Re x')
plt.show()

#
spos=[0.01, 0.05333333333333334, 0.09666666666666666, 0.14, 0.18333333333333335, 0.22666666666666668, 0.27, 0.31333333333333335, 0.3566666666666667, 0.4]
conv30=[0.0348, 0.0467, 0.0493, 0.0851, 0.0878, 0.0935, 0.0956, 0.1026, 0.1039, 0.126]
conv60=[0.0487, 0.0508, 0.065, 0.0738, 0.0742, 0.0796, 0.0858, 0.0867, 0.095, 0.097]
conv120=[0.0478, 0.051, 0.0638, 0.0724, 0.0754, 0.0806, 0.0838, 0.0872, 0.0938, 0.0985]
conv240=[0.0475, 0.0511, 0.0633, 0.0714, 0.0748, 0.0813, 0.0833, 0.0873, 0.093, 0.0994]
fig = plt.subplots()
plt.plot(spos,conv30,'-b',marker='.',ls='--',label='res30')
plt.plot(spos,conv60,'-r',marker='^',ls='-.',label='res60')
plt.plot(spos,conv120,'-g',marker='o',ls=':',label='res120')
plt.plot(spos,conv240,'m',marker='v',label='res240')
plt.xlabel('Source position')
plt.ylabel('Flux ratio')
plt.title('Flux ratio for 1 μ wide and high pyramid, varying source position, Y-polarized ')
plt.legend(loc='best')
plt.show()
##fftot 7.64551521
conv30b=[0.3162, 0.5302, 0.5403, 0.591, 0.6078, 0.6119, 0.6398, 0.8071, 0.8192, 0.8259]
conv60b= [0.2554, 0.4451, 0.467, 0.4975, 0.5089, 0.5135, 0.5444, 0.6187, 0.63, 0.634]
conv120b=[0.2487, 0.4241, 0.4639, 0.473, 0.4901, 0.5036, 0.541, 0.6041, 0.6151, 0.6261]
conv240b=[0.2454, 0.4193, 0.4626, 0.4685, 0.4845, 0.5001, 0.5388, 0.5995, 0.6113, 0.6231]
plt.plot(spos,conv30b,'-b',marker='.',ls='--',label='res30')
plt.plot(spos,conv60b,'-r',marker='^',ls='-.',label='res60')
plt.plot(spos,conv120b,'-g',marker='o',ls=':',label='res120')
plt.plot(spos,conv240b,'m',marker='v',label='res240')
plt.xlabel('Source position')
plt.ylabel('Flux ratio')
plt.title('Flux ratio for 1 μ wide and high pyramid, varying source position, Y-polarized ')
plt.legend(loc='best')
plt.show()




