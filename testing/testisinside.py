import meep as mp
import math
air = [1, 1, 1]
GaN = [5.76, 5.76, 5.76]
ph = 1
pw = 1
sh = 0
#120 ([2.08620994702761], [0.15872501266542283], [0.07608295267289424])
#60 ([2.0893500586301563], [0.15725656619367934], [0.07526578207616473])
sz=100


def Insidexy(vec):
    while (vec[2] <= 5 and vec[2] >= 0):
        h = pw/(2*ph)*vec[2]-(pw/(2*ph))*(sz/2-sh-ph)
        print(h)
        v = h*math.tan(math.pi/6)
        center = mp.Vector3(h, v, 0)
        # transform the test point locally and to quadrant 2m nm
        q2x = math.fabs(vec[0] - center.x+h)
        # transform the test point locally and to quadrant 2
        q2y = math.fabs(vec[1] - center.y+v)
        if (q2x > h) or (q2y > v*2):
            return air
        if ((2*v*h - v*q2x - h*q2y) >= 0): #finally the dot product can be reduced to this due to the hexagon symmetry
            return GaN
        else:
            return air 
    else:
        return [0,0,0]
		#paints out a hexagon with the help of 4 straight lines in the big if statement
def Insidexy2(vec):
	while (vec[2] <= 5 and vec[2] >=0):
		v=(pw/(2*ph))*vec[2]+(pw/(2*ph))*(sh+ph-sz/2)
		h=math.cos(math.pi/6)*v
		k=1/(2*math.cos(math.pi/6))
		if (-h<=vec[0]<=h and vec[1] <= k*vec[0]+v and vec[1] <= -k*vec[0]+v and vec[1] >= k*vec[0]-v and vec[1] >= -k*vec[0]-v):
		    return GaN
		else:
			return air
	else:
		return [0,0,0]
mat1=[]
mat2=[]
for x in range(-50,50,1):
    for y in range(-50,50,1):
        for z in range(-50,50,1):
            vec = [x,y,z]
           # print(vec)
            mat1.append(Insidexy(vec))
            mat2.append(Insidexy2(vec))
#print(mat1)
#print(mat2)
if mat1 == mat2:
    print('they work the same')
else:
    print('they do not work the same')
##
