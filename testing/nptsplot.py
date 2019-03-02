import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math			

#time: 5 dpml: 0.5 res: 40 offset: 0 r; 20
#symmetry=[mp.Mirror(mp.X),mp.Mirror(mp.Y,phase=-1),mp.Mirror(mp.Z)]
NPTS=[30,60,90,120,150,200,250,300,400,450,600,700,800,900,1000,1200,1400,2000,3000]
farfieldflux=[0.44217483,0.43487465,0.43771086,0.43951345,0.4367545,0.43798811,0.43888323,0.43973498,0.43889659,0.43792807,
0.43707977,0.43795148,0.4385902,0.43843797,0.43762474,0.43794102,0.43830161,0.43786075,0.43794555]
time=[2.5,5,10,20,40]
total_flux=[0.7600748646240835,1.325014335273617,1.3251745134242714,1.3251602765723127,1.3251661436561768]

#time: 5 dpml: 0.5
total_flux=[1.2896091139499157,1.3203689503454936,1.325014335273617,1.3242840037511816,1.3249920087996594,1.3252489988563414]
res=[10,20,40,80,120,160]
plt.title('flux vs res time:5')
plt.plot(res,total_flux)
plt.show()

#time:5 dpml: 0.5 res: 80 r:20 flux vs npts
total_flux=[0.63580136, 0.62073483,0.62733091,0.62863217,0.62745988,0.62625614,0.62664326]
npts=[30, 100, 200, 400, 800, 1600, 3200]
plt.title('flux vs npts')
plt.plot(npts,total_flux)
plt.show()

#simulation_time: 15 dpml: 0.1 r: 300 npts: 1600
resolution=[10,20,40,60,80,120,140]
ratio=[0.16217052,0.04564019,0.03975171,0.04212829,0.04469922,0.04801199,0.04970425]
plt.title('res vs ratio')
plt.plot(resolution,ratio)
plt.show()
