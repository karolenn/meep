import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math			

angle=[math.pi, math.pi/1.3,math.pi/1.6,math.pi/2,math.pi/2.5,math.pi/3,math.pi/4,math.pi/5,math.pi/6,math.pi/7,math.pi/8,math.pi/10,math.pi/12,math.pi/14]
flux_rat=[1.0035918043747372, 0.988070219532373, 0.7602468168436849, 0.5033597777515132, 0.3066661696277227, 0.19642462098998423, 0.09275944372960104, 0.05042406850837538, 0.030227995387703557, 0.019530452508231693, 0.013359264989984297, 0.0070111518769213925, 0.004109188259857358,0.0026129728783640307]


Pr=[0.006678231526178049, 0.019992779538939475, 0.03325154483786669, 0.0464395101520752, 0.059575262334436024, 0.07261503702651052, 0.08561215061325103, 0.09849172086060741, 0.11132423495620121, 0.12404787498965879, 0.1366768377452042, 0.1492646241133075, 0.16164326819924577, 0.17412223917104058, 0.1862085807128464, 0.1985984389395802, 0.21037102874437846, 0.22266992075434053, 0.2341402943221975, 0.2463163938310097, 0.257532266313835, 0.2695249076421314, 0.2805612988970082, 0.2922922908733852, 0.3032324243205821, 0.3146249714768197, 0.3255370870642878, 0.33653718766856683, 0.34745534930619243, 0.35804924093893487, 0.3689646644354011, 0.3791864186506022, 0.3900514541871397, 0.3999773031226031, 0.41071942853662746, 0.420449150909367, 0.4309898416186659, 0.4406194261859984, 0.45089325739282016, 0.46048624372284447, 0.4704571942031597, 0.48002387978443484, 0.4896962619105188, 0.4991891217944278, 0.508609894619165, 0.5179386809370887, 0.5271884642232967, 0.5362502108757541, 0.5454236212360144, 0.5541355214279107, 0.5633156500146022, 0.5716381280191064, 0.580871616370228, 0.5888162505028302, 0.5980937697120517, 0.6057203228132544, 0.6149654978522449, 0.6223757876021729, 0.6314464916674029, 0.6387775192199185, 0.6474847250770346, 0.6548958396279475, 0.6630414527927105, 0.6706896506272318, 0.6781148151495076, 0.6861205463054305, 0.6927469183224191, 0.7011618630208014, 0.7070099187992599, 0.7157983085193356, 0.7209803458266434, 0.7300162009161529, 0.7347167529999681, 0.7437909806092279, 0.7482501578420204, 0.7570831461767493, 0.761585817506498, 0.7698504679510499, 0.7747082932723307, 0.7820728585579452, 0.7875839000763247, 0.7937745729079403, 0.8001616834199711, 0.8050272338989454, 0.8123780186923814, 0.8159297342124836, 0.8241663467938105, 0.8265779574800174, 0.8354666040109391, 0.8370437441700803, 0.8462278328444898, 0.847372517161988, 0.8564058140437524, 0.857591513552881, 0.8659666669099022, 0.8677116745209354, 0.8749043894418365, 0.8777145529744825, 0.8832641613194498, 0.8875343608081486, 0.8911498126547919, 0.8970566143739784, 0.8987010032548439, 0.9061446712797454, 0.9060508748303546, 0.914679801111969, 0.9132934902688938, 0.9225851795883575, 0.9204814286557143, 0.9298194207896945, 0.9276448455629407, 0.9363581532324261, 0.934802617013908, 0.9421958693376008, 0.9419449561747327, 0.9473755547789685, 0.9489990358479053, 0.9520151166584745, 0.9558150232272952, 0.9562924931238538, 0.9622003686307915, 0.9603886007095597, 0.9679869366417119, 0.964430101908091, 0.9730795681232802, 0.9684749914253458, 0.9774488701994579, 0.9725420324831714, 0.981087077086471, 0.9766436050365566, 0.9839841014554327, 0.9807807761899021, 0.9861564109328397, 0.984900008472007, 0.9876981733893557, 0.9888560927134382, 0.9887920111361584, 0.9924312321029866, 0.9896537837658475, 0.9954128261332054, 0.9904528131722813, 0.9976727097325973, 0.9912730717444559, 0.9991822374913989, 0.9921386694577106, 0.9999577533290381, 0.9930646397422559, 1.0, 0.9940719075948955, 0.9992944098949392, 0.995142226147182, 0.9978718316556091, 0.9961527862674209, 0.995862111314704, 0.9968643062860193, 0.9934780364488822, 0.9970011319932812, 0.9909388326132288, 0.9963767844306646, 0.9884019582369147, 0.9949628183521694, 0.9859567534423256, 0.992839653577808, 0.9836640416539069, 0.9900760464502936, 0.9815754568721338, 0.9866518800270135, 0.9796857540249218, 0.9824957604887464, 0.9778473097113999, 0.9775968500583362, 0.9757410762705476, 0.9720829464122046, 0.9729791616741946, 0.9661998075171144, 0.9693021288670337, 0.9602255999588613, 0.964722426463579, 0.9543992643217595, 0.9594795850269074, 0.9488952477480812, 0.9538269495941489, 0.9438063537737122, 0.9478233638682354, 0.9390873368599754, 0.9413023779827866, 0.9344797320196048, 0.9340345770978864, 0.9295131080537378, 0.925937883772347, 0.9236637546261216, 0.9171739665739418, 0.9166243223038983, 0.9080889414649663, 0.9084997328864413, 0.8990747143830796, 0.8997508816861917, 0.8904397871602817, 0.8908931477080649, 0.8823163392892629, 0.8821638452227885, 0.8745921956456282, 0.8734029107975178, 0.8668849117058217, 0.8642153308983993, 0.8586187873769207, 0.8542673932612569, 0.849240524243292, 0.8434980086961066, 0.8384991969035943, 0.8321318060568816, 0.8266140353197096, 0.8205350139022372, 0.8141834196853478, 0.8090265729987373, 0.8018632508163036, 0.7977316372999893, 0.7900202764948102, 0.7865151072828764, 0.7785841304539484, 0.7750110710950232, 0.7671667561426241, 0.7627548314529906, 0.7553300903155562, 0.749388527832267, 0.7428117381804269, 0.7348520363705042, 0.7295915161285907, 0.7194446069619845, 0.715808745421068, 0.7037007333907745, 0.7016187341695073, 0.6881416173562912, 0.6870780126435899, 0.673048101683232, 0.6721069652720196, 0.6583797449953681, 0.6565353766415228, 0.6438590413546282, 0.6402035440709083, 0.6291387719000224, 0.6230682131543818, 0.6139442856431859, 0.6052563577912892, 0.5981297282672434, 0.5870331851800143, 0.5816551773371208, 0.5686987724365774, 0.5645312129524577, 0.5504703753395779, 0.5467762064615349, 0.5324126600475316, 0.5284065216869513, 0.5144429974546224, 0.5094530732449476, 0.4963927821321192, 0.4899825111715699, 0.47808168872184115, 0.4701016312775607, 0.4593695232777621, 0.44993678526981673, 0.4401748072889867, 0.42959761362072574, 0.4204694730476155, 0.40914501457097596, 0.40026486100524417, 0.38858003902527416, 0.3795986776028031, 0.36785673856300005, 0.35852446969055424, 0.3469086465816806, 0.3371010796869862, 0.3256742870752671, 0.315380765778601, 0.3041121540509029, 0.29339832284025835, 0.282203788772751, 0.2711655695910271, 0.25994889714923414, 0.24867385143858836, 0.23735731168609694, 0.22590300424271087, 0.2144410512551817, 0.20283187133490002, 0.19120823419908176, 0.17944539405567111, 0.16765994672090676, 0.1557361190687503, 0.1437906522931239, 0.1317013013800916, 0.11959159407640882, 0.10733856470930216, 0.09505518360091501, 0.08264293179449304, 0.07017772497966482, 0.05760678099863492, 0.04495880183696872, 0.03222274346062947, 0.01939781272712362, 0.006488056232566634]


x=np.linspace(0,math.pi,300)
print(len(flux_rat))
#plt.plot(x,Pr)
#plt.plot(x,np.sin(x/2)**2)
#plt.plot(angle,flux_rat)
#plt.show()

##fftot 7.64551521
angleTheta=math.pi
#npts=80
xpts=[]
ypts=[]
zpts=[]
def fibspherepts(r,theta,npts,xpts,ypts,zpts):

	offset=0.75/npts
	range_npts=int((theta/math.pi)*npts)
	increment = math.pi*(3 - math.sqrt(5))

	for n in range(range_npts):
		zpts.append(r*((n*offset-1)+(offset/2)))
		R = r*math.sqrt(1-pow(zpts[n]/r,2))
		phi = (n % npts)*increment
		xpts.append(R*math.cos(phi))
		ypts.append(R*math.sin(phi))

	return(xpts,ypts,zpts)

r=100
theta=math.pi/4
phi=math.pi*2
nPts=16
npts=40**2
npts=1600
xPts=[]
yPts=[]
zPts=[]


def sphericalpts(r,theta,phi,nPts,xPts,yPts,zPts):

	for n in range(nPts):
		angleTheta=(n/nPts)*theta
		for m in range(nPts):
			anglePhi=(m/nPts)*phi
			xPts.append(r*math.sin(angleTheta)*math.cos(anglePhi))
			yPts.append(r*math.sin(angleTheta)*math.sin(anglePhi))
			zPts.append(-r*math.cos(angleTheta))

	return(xPts,yPts,zPts)
fibspherepts(r,theta,npts,xpts,ypts,zpts)
sphericalpts(r,theta,phi,nPts,xPts,yPts,zPts)
print(len(xpts))
print(int((theta/math.pi)*npts))
#print(xpts)
#plt.xlim(-4,4)
#plt.ylim(-4,4)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(xpts,ypts,zpts,color='r')
plt.gca().set_aspect('equal', adjustable='box')
ax.scatter3D(xPts,yPts,zPts)
ax.set_zlim(-100,100)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()



