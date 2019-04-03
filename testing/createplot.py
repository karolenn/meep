import meep as mp	
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math			



#def plotdata(ff, height, width, position, freq):
data= np.loadtxt("pyramid.out",delimiter=' ',dtype=list,usecols=(5,13,15,17,19))
lendata=len(data)
ff_array = []
print(data)
for i in range(lendata):
	ff_array.append(data[i,0])

ff_arrayD = []
print(ff_array)
print(ff_array[1])
ff_array1=""
ff_array1=ff_array[1].replace("[","",1)
ff_array1=ff_array1.replace("]","",1)
print(ff_array1)
print(type(ff_array1))
ff_arrayD.append(float(ff_array1))
lenff = len(ff_array)
	
	#for i in range(lenff):
		#lenff.append(double(


#insert (x,y) long array of sim data output and output array of strings of given argument
def getarray(data,arg):
	array = []
	if arg=='s_pos':
		j=0
	if arg=='height':
		j=1
	if arg=='width':
		j=2
	if arg=='freq':
		j=3
		
	
	for i in range(len(data)):
		array.append(data[i,j])

	return(array)

#insert (1,x) long array with strings (containing [],) and convert them to array of doubles
def convert2double(narray):
	lenarray=len(narray)
	Darray=[]
	Sarray=""
	for i in range(lenarray):
		Sarray=narray[i]
		Sarray=Sarray.replace("[","",1)
		Sarray=Sarray.replace("]","",1)
		Darray.append(float(Sarray))
	return(Darray)

#def plotarray(Ynarray,Xnarray,labely,labelx)
	
		




