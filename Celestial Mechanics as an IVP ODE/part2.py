import diffeq_2 as de

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
rc('text', usetex=True)


# Differential equation in 1de standard form
def f1(y, t):
	return np.array([y[2], y[3], -(y[0] - a*np.cos(8**(1/2)*t))/((y[0] - a*np.cos(8**(1/2)*t))**2 + (y[1] - a*np.sin(8**(1/2)*t))**2)**(3/2)
		-(y[0] + a*np.cos(8**(1/2)*t))/((y[0] + a*np.cos(8**(1/2)*t))**2 + (y[1] + a*np.sin(8**(1/2)*t))**2)**(3/2),
		-(y[1] - a*np.sin(8**(1/2)*t))/((y[0] - a*np.cos(8**(1/2)*t))**2 + (y[1] - a*np.sin(8**(1/2)*t))**2)**(3/2)
		-(y[1] + a*np.sin(8**(1/2)*t))/((y[0] + a*np.cos(8**(1/2)*t))**2 + (y[1] + a*np.sin(8**(1/2)*t))**2)**(3/2)], dtype=np.float64)

# Differential equation in 2de standard form
def f2(y):
	return np.array([-y[0]/(y[0]**2 + y[1]**2)**(3/2),
		-y[1]/(y[0]**2 + y[1]**2)**(3/2)], dtype=np.float64)


# Energy of system
def erg(y):
	return 1/2 * (y[2]**2 + y[3]**2) - (y[0]**2 + y[1]**2)**(-1/2)

# Angular momentum of system
def angmom(y):
	return y[0]*y[3] - y[1]*y[2]

# Runge-Lenz vector of system
def rnglnz(y, comp):
	L = angmom(y)
	if comp == 0:
		return y[3]*L -y[0]/(y[0]**2 + y[1]**2)**(1/2)
	elif comp == 1:
		return -y[2]*L -y[1]/(y[0]**2 + y[1]**2)**(1/2)
	else:
		print('Not a valid component number')
		return None

# Distance from center
def dist(y, cent=(0, 0)):
	return ((y[0] - cent[0])**2 + (y[1] - cent[1])**2)**(1/2)

# Semi-major axis of orbit
def semimaj(v0):
	return 1/(2 - v0**2)

# Exact period 
def T_exact(v0):
	a = semimaj(v0)
	if a < 0:
		print("Unbounded orbits don't have a well-defined period")
		return None
	else:
		return 2 * np.pi * (2 - v0**2)**(-3/2)
T_exact = np.vectorize(T_exact)

# Find x coordinates of points after one orbit
def rtrnpts(y):
	points = []
	for i in range(len(y)-1):
		if y[i][1]*y[i+1][1] < 0 and y[i][0] > 0:
			x = y[i][0] - y[i][1]*(y[i+1][0] - y[i][0])/(y[i+1][1] - y[i][1])
			points.append([i,x])

	return np.array(points)

# Periods and their times for the orbit
def pers(y, t):
	times = [[0, 0]]
	for i in range(len(y)-1):
		if y[i][1]*y[i+1][1] < 0 and y[i][0] > 0:
			x = y[i][0] - y[i][1]*(y[i+1][0] - y[i][0])/(y[i+1][1] - y[i][1])
			T = t[i] + (t[i+1] - t[i]) * dist(y[i], [x, 0])/dist(y[i+1], y[i])
			times.append([T, T - times[-1][0]])

	return np.array(times[1:])


# Setup for RKF
hmax = 0.7
hmin = 10**(-16)
tol = 3.5*10**(-5)

# Intial conditions
a = 0.27
T = 50

# Solve the differential equation forwards
t, y = de.rkf(f1, 0, T, [0, 1, 1, 0], tol, hmax, hmin)


plt.title(fr'Orbit with $a={a}$')
plt.xlabel(r'$x$ coordinate')
plt.ylabel(r'$y$ coordinate')



plt.plot(np.linspace(-a, a), [(a**2 - x**2)**(1/2) for x in np.linspace(-a, a)], 'r-', label='Binary')
plt.plot(np.linspace(-a, a), [-(a**2 - x**2)**(1/2) for x in np.linspace(-a, a)], 'r-')
plt.plot(y[:,0], y[:,1], label='Planet')

plt.legend()
#plt.savefig('images/binary_prec.png')
plt.show()
