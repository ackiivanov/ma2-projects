import diffeq_2 as de

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
rc('text', usetex=True)


# Parameters of our system
v0 = 1
phi0 = 1


# Differential equation in 1de standard form
def f1(y, t):
	return np.array([y[2], y[3], -y[0]/(y[0]**2 + y[1]**2)**(3/2) -
		(y[0] + 10 - 2*np.abs(v0)*t)/((y[0] + 10 - 2*np.abs(v0)*t)**2 + (y[1] - 1.5)**2)**(3/2),
		-y[1]/(y[0]**2 + y[1]**2)**(3/2) -
		(y[1] -1.5)/((y[0] + 10 - 2*np.abs(v0)*t)**2 + (y[1] - 1.5)**2)**(3/2)], dtype=np.float64)

# Differential equation in 2de standard form
def f2(y, t):
	return np.array([-y[0]/(y[0]**2 + y[1]**2)**(3/2) -
		(y[0] + 10 - 2*np.abs(v0)*t)/((y[0] + 10 - 2*np.abs(v0)*t)**2 + (y[1] - 1.5)**2)**(3/2),
		-y[1]/(y[0]**2 + y[1]**2)**(3/2) -
		(y[1] -1.5)/((y[0] + 10 - 2*np.abs(v0)*t)**2 + (y[1] - 1.5)**2)**(3/2)], dtype=np.float64)


# Energy of system
def erg1(y, t):
	return (1/2 * (y[2]**2 + y[3]**2) - (y[0]**2 + y[1]**2)**(-1/2) - ((y[0] + 10 - 2*np.abs(v0)*t)**2 + (y[1] - 1.5)**2)**(-1/2))

# Energy in other frame
def erg2(y, t):
	return (1/2 * ((y[2] - 2*np.abs(v0))**2 + y[3]**2) - (y[0]**2 + y[1]**2)**(-1/2) - ((y[0] + 10 - 2*np.abs(v0)*t)**2 + (y[1] - 1.5)**2)**(-1/2))


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

"""
N = 1000
print(f'Step size: h = {10/np.abs(v0)/N:.8f}')
t = np.linspace(0, 10*10/np.abs(v0), 10*N)

phi0_lst = np.linspace(0, 2*np.pi, 50)

ergs = []
for phi0 in phi0_lst:
	print(phi0)
	y = de.pefrlt(f2, [np.cos(phi0), np.sin(phi0)], [-v0*np.sin(phi0), v0*np.cos(phi0)], t)

	plt.plot(0, 0, 'ro')
	plt.axhline(y=1.5, color='g')
	plt.plot(y[:,0], y[:,1])
	plt.show()

	ergs.append(erg1(y[-1], t[-1]))

plt.plot(phi0_lst, ergs)

plt.show()
"""

"""
# Setup for RKF
hmax = 1
hmin = 10**(-14)
tol = 10**(-7)

# List of initial angles
phi0_min = 0
phi0_max = 2*np.pi
phi0_num = 50
phi0_lst = np.linspace(phi0_min, phi0_max, phi0_num)

# List of initial velocites
v0_amp = 2.0
v0_num = 4
v0_lst = v0_amp * np.linspace(-1, 1, v0_num)

# Calculate what happens to planet
fate = np.zeros((v0_num, phi0_num))
for i in range(len(v0_lst)):
	for j in range(len(phi0_lst)):
		v0 = v0_lst[i]

		# Indicator
		print(v0_lst[i],phi0_lst[j])

		# Solve the differential equation
		t, y = de.rkf(f1, 0, 7*10/np.abs(v0_lst[i]), [np.cos(phi0_lst[j]),
			np.sin(phi0_lst[j]), -v0_lst[i]*np.sin(phi0_lst[j]),
			v0_lst[i]*np.cos(phi0_lst[j])], tol, hmax, hmin)

		# Check what happens
		if erg1(y[-1], t[-1]) >= 0:
			fate[i][j] = 0
		elif dist(y[-1]) <= dist(y[-1], [-10 + 2*np.abs(v0_lst[i])*t[-1], 1.5]):
			fate[i][j] = 1
		else:
			fate[i][j] = 2

# Plotting
fig, ax = plt.subplots()

ax.set_title(fr'Parameter Space Portrait with Fates of Planet')
ax.set_xlabel(r'Initial velocity $v_0$')
ax.set_ylabel(r'Initial angle $\phi_0$')
ax.grid()

ax.contourf(phi0_lst, v0_lst, fate, 250, alpha=0.7, antialiased=True)

plt.savefig('images/parameter_space.png')
plt.show()
"""


# Setup for RKF
hmax = 0.7
hmin = 10**(-16)
tol = 3.5*10**(-5)

# List of initial angles
phi0_min = 0
phi0_max = 2*np.pi
phi0_num = 100
phi0_lst = np.linspace(phi0_min, phi0_max, phi0_num)

# Calculate what happens to planet
ergs = [[], [], [], []]
ergdiffs = [[], [], [], []]
for phi0 in phi0_lst:
	
	# Indicator
	print(v0, phi0)

	# Solve the differential equation forwards
	tf, yf = de.rkf(f1, 0, 5*10/np.abs(v0), [np.cos(phi0), np.sin(phi0),
		-v0*np.sin(phi0), v0*np.cos(phi0)], tol, hmax, hmin)

	# Solve the differential equation backwards
	tb, yb = de.rkf(f1, 0, 5*10/np.abs(v0), [np.cos(phi0), np.sin(phi0),
		v0*np.sin(phi0), -v0*np.cos(phi0)], tol, hmax, hmin)

	# Calculate final energy
	ergs[0].append(erg1(yf[-1], tf[-1]))
	ergs[1].append(erg1(yb[-1], tb[-1]))

	ergs[2].append(erg2(yf[-1], tf[-1]))
	ergs[3].append(erg2(yb[-1], tb[-1]))

	# Calculate energy change over time
	ergdiffs[0].append(np.abs(erg1(yf[-1], tf[-1]) - erg1(yf[-2], tf[-2]))/(tf[-1] - tf[-2]))
	ergdiffs[1].append(np.abs(erg1(yb[-1], tb[-1]) - erg1(yb[-2], tb[-2]))/(tb[-1] - tb[-2]))
	
	ergdiffs[2].append(np.abs(erg2(yf[-1], tf[-1]) - erg2(yf[-2], tf[-2]))/(tf[-1] - tf[-2]))
	ergdiffs[3].append(np.abs(erg2(yb[-1], tb[-1]) - erg2(yb[-2], tb[-2]))/(tb[-1] - tb[-2]))


	if 5.46 <= phi0 <= 5.7:
		plt.title(fr'Orbit')
		plt.xlabel(r'$x$ coordinate')
		plt.ylabel(r'$y$ coordinate')
		plt.plot(yb[:,0], yb[:,1], label=fr'$v_0={v0:.3f}$')

plt.title(fr'Orbits for Passing Star Capture')
plt.xlabel(r'$x$ coordinate')
plt.ylabel(r'$y$ coordinate')
plt.plot(0, 0, 'ro', label='Massive Star')
plt.axhline(y=1.5, color='k', label='Passing Star')
plt.grid()
plt.legend()
plt.savefig('images/around2.png')
plt.show()


# Plotting
fig, ax = plt.subplots()

ax.set_title(fr'Energy as a Function of Initial Angle, Frame 1, $v_0={v0}$')
ax.set_xlabel(r'Initial angle $\phi_0$')
ax.set_ylabel(r'Energy $E$')
ax.grid()

ax.axhline(y=0, color='k', linestyle='-')
ax.plot(phi0_lst, ergs[0], label='Forwards')
ax.plot(phi0_lst, ergs[1], label='Backwards')

axin = fig.add_axes([0.34, 0.5, 0.37, 0.3])
axin.set_title(r'$|\mathrm{d} E/\mathrm{d} t|$, log scale')
axin.set_yscale('log')
axin.grid()

axin.plot(phi0_lst, ergdiffs[0])
axin.plot(phi0_lst, ergdiffs[1])

ax.legend(loc='best')
#plt.savefig('images/energy1_phi0.png')
plt.show()


# Plotting
fig, ax = plt.subplots()

ax.set_title(fr'Energy as a Function of Initial Angle, Frame 2, $v_0={v0}$')
ax.set_xlabel(r'Initial angle $\phi_0$')
ax.set_ylabel(r'Energy $E$')
ax.set_ylim(-1.6, 6.5)
ax.set_xlim(-0.5, 6.3)
ax.grid()

ax.axhline(y=0, color='k', linestyle='-')
ax.plot(phi0_lst, ergs[2], label='Forwards')
ax.plot(phi0_lst, ergs[3], label='Backwards')

ax.plot([5.29, 5.96, 5.96, 5.29, 5.29], [0.34, 0.34, -0.44, -0.44, 0.34], 'r-')

axin = fig.add_axes([0.19, 0.49, 0.37, 0.31])
axin.set_title(r'$|\mathrm{d} E/\mathrm{d} t|$, log scale')
axin.set_yscale('log')
axin.grid()

axin.plot(phi0_lst, ergdiffs[2])
axin.plot(phi0_lst, ergdiffs[3])


ax.legend(loc='best')
#plt.savefig('images/energy2_phi0.png')
plt.show()