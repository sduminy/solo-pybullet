from math import sin, cos, pi
import numpy as np

from IPython import embed

"""
T = 0.4	#period of 0.4s

dx = 0.04
dz = 0.06

def ftraj1(DeltaT, x0, z0):	#arguments : time, initial position x and z
		global T, dx, dz
		x = []
		z = []
		for t in DeltaT:
			if t >= T:
				t %= T
			x.append(x0-dx*np.cos(2*np.pi*t/T))
			if t <= T/2.:
				z.append(z0+dz*np.sin(2*np.pi*t/T))
			else:
				z.append(0)
		return np.matrix([x,z])
"""

from numpy import matrix
from curves import bezier

import matplotlib.pylab as plt

N_DISCRETIZATION = 100

T = 0.3				# period of one gait cycle
L = 0.04			# desired leg stride in a gait cycle
Tst = T*.5			# time of the stance phase
Tsw = T*.5			# time of the swing phase
Vdesired = L/Tst	# desired velocity of the robot
vD = 0.001			# virtual displacement to ensure the body stability
h = 0.06			# maximal height of the foot

xF0 = 0.19	#initial position of the front feet
xH0 = -0.19	#initial position of the hind feet
z0 = 0.0	#initial altitude of each foot

N_SIMULATION = 100 	# Number of simulation steps


def ftraj(DeltaT, x0, z0):
	
	# Bezier curves for the swing phase

	P0  = [    x0    ,   z0    ]
	P1  = [x0-0.125*L,   z0    ]
	P2  = [ x0-0.5*L , z0+0.8*h]
	P3  = [ x0-0.5*L , z0+0.8*h]
	P4  = [ x0-0.5*L , z0+0.8*h]
	P5  = [ x0+0.5*L , z0+0.8*h]
	P6  = [ x0+0.5*L , z0+0.8*h]
	P7  = [ x0+0.5*L ,  z0+h   ]
	P8  = [ x0+1.5*L ,  z0+h   ]
	P9  = [ x0+1.5*L ,  z0+h   ]
	P10 = [x0+1.125*L,   z0    ]
	P11 = [  x0 + L  ,   z0    ]

	waypoints = matrix([P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11]).T
	bc = bezier(waypoints, 0.0, Tsw)

	x = []
	z = []
	
	for t in (DeltaT):
		t %= T
		if t <= Tsw :
			x.append(bc(t)[0,0])
			z.append(bc(t)[1,0])
			
		else :
			t -= Tsw
			x.append(P11[0] - Vdesired * t)
			z.append(-vD * cos(pi * (0.5 - Vdesired * t / L)) - P0[1])
			
	return x,z

DeltaT = np.linspace(0., T, N_SIMULATION)

X,Z = ftraj(DeltaT, xF0, z0)

plt.figure(2)
plt.plot(X,Z)
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title("Foot trajectory using Bezier curve")
plt.show()

"""
DeltaT = np.linspace(0,2*T,300)
x0 = -0.19
z0 = 0
X = np.array(ftraj2(DeltaT, x0, z0)[0].T)
Z = np.array(ftraj2(DeltaT, x0, z0)[1].T)
plt.plot(X,Z)
plt.xlabel('displacement by x (m)')
plt.ylabel('displacement by z (m)')
plt.grid()
plt.title('Parametric curve of the trajectory')
plt.show()
"""
embed()
