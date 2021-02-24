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
	return np.array([y[2], y[3], -y[0]/(y[0]**2 + y[1]**2)**(3/2),
		-y[1]/(y[0]**2 + y[1]**2)**(3/2)], dtype=np.float64)

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


"""
# Calibration with circular orbit

# Parameters of our system
v0 = 1

# Step length and number
h = 0.1
N = 5000
t = np.linspace(0, N*h, N)

# Calculate orbits
y_rku4 = de.rku4(f1, [1, 0, 0, v0], t)
y_verl = de.verlet(f2, [1, 0], [0, v0], t)
y_pefrl = de.pefrl(f2, [1, 0], [0, v0], t)

# Calculate periods
pers_rku4 = pers(y_rku4, t)
pers_pefrl = pers(y_pefrl, t)
pers_verl = pers(y_verl, t)


# Plotting
fig, ax = plt.subplots()

ax.set_title(fr'Distance from Star over Time, $v_0={v0}$, $h={h}$, $N=200$')
ax.set_xlabel(r'Time $t$')
ax.set_ylabel(r'Relative deviation $\delta r / r$')
ax.set_yscale('log')
ax.grid()

ax.plot(t[0:1000], np.abs(dist(y_rku4[0:1000].T) - 1), label='RK4')
ax.plot(t[0:1000], np.abs(dist(y_pefrl[0:1000].T) - 1), label='PEFRL')
ax.plot(t[0:1000], np.abs(dist(y_verl[0:1000].T) - 1), label='Verlet')

ax.legend(loc='upper left')
plt.savefig('images/circular_dist_log.png')
plt.show()


# Plotting
fig, axs = plt.subplots(2, 2, sharey=False, sharex=False)

# Plot of orbit
ax = axs[0, 0]
ax.set_title('Orbit')
ax.set_xlabel(r'$x$ coordinate')
ax.set_ylabel(r'$y$ coordinate')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.05, 1.05)
ax.grid()

ax.plot(0, 0, 'ro')
ax.plot(y_rku4[:,0], y_rku4[:,1], label='RK4')
ax.plot(y_pefrl[:,0], y_pefrl[:,1], label='PEFRL')
ax.plot(y_verl[:,0], y_verl[:,1], label='Verlet')

# Plot of energy deviations
ax = axs[0, 1]
ax.set_title('Energy deviations')
ax.set_xlabel(r'Time $t$')
ax.set_ylabel(r'Relative deviation $|\delta E/E|$')
ax.set_ylim(-6*10**(-6), 0.0001)
ax.grid()

ax.plot(t, np.abs(erg(y_rku4.T)/erg([1, 0, 0, v0]) - 1))
ax.plot(t, np.abs(erg(y_pefrl.T)/erg([1, 0, 0, v0]) - 1))
ax.plot(t, np.abs(erg(y_verl.T)/erg([1, 0, 0, v0]) - 1))

ax01in = fig.add_axes([0.74, 0.66, 0.15, 0.15])
ax01in.set_title('log scale')
ax01in.set_yscale('log')
ax01in.grid()

ax01in.plot(t[N-200:N], np.abs(erg(y_rku4[N-200:N].T)/erg([1, 0, 0, v0]) - 1))
ax01in.plot(t[N-200:N], np.abs(erg(y_pefrl[N-200:N].T)/erg([1, 0, 0, v0]) - 1), color=u'#ff7f0e')
ax01in.plot(t[N-200:N], np.abs(erg(y_verl[N-200:N].T)/erg([1, 0, 0, v0]) - 1), color=u'#2ca02c')

# Plot of angular momentum deviations
ax = axs[1, 0]
ax.set_title('Angular momentum deviations')
ax.set_xlabel(r'Time $t$')
ax.set_ylabel(r'Relative deviation $|\delta L/L|$')
ax.set_ylim(-3.5*10**(-6), 5*10**(-5))
ax.grid()

ax.plot(t, np.abs(angmom(y_rku4.T)/angmom([1, 0, 0, v0]) - 1))
ax.plot(t, np.abs(angmom(y_pefrl.T)/angmom([1, 0, 0, v0]) - 1))
ax.plot(t, np.abs(angmom(y_verl.T)/angmom([1, 0, 0, v0]) - 1))

ax10in = fig.add_axes([0.31, 0.17, 0.15, 0.15])
ax10in.grid()

ax10in.plot(t[N-100:N], np.abs(angmom(y_pefrl[N-100:N].T)/angmom([1, 0, 0, v0]) - 1), color=u'#ff7f0e')
ax10in.plot(t[N-100:N], np.abs(angmom(y_verl[N-100:N].T)/angmom([1, 0, 0, v0]) - 1), color=u'#2ca02c')

# Period deviations
ax = axs[1, 1]
ax.set_title('Period deviations')
ax.set_xlabel(r'End time of period $t$')
ax.set_ylabel(r'Relative deviation $|\delta T|/T$')
ax.set_yscale('log')
ax.grid()

ax.plot(pers_rku4[:,0], np.abs(pers_rku4[:,1]/(2*np.pi) - 1))
ax.plot(pers_pefrl[:,0], np.abs(pers_pefrl[:,1]/(2*np.pi) - 1))
ax.plot(pers_verl[:,0], np.abs(pers_verl[:,1]/(2*np.pi) - 1))


fig.suptitle(fr'Calculated orbit, $v_0 = {v0}$, $h = {h}$, $N = {N}$')
axs[0,0].legend(loc='upper left')
plt.gcf().set_size_inches(13.5, 9)
plt.savefig('images/circular.png')
plt.show()
"""

"""
# Shape of orbit for different v0
v0_lst = [0.7, 1, 1.2, 2**(1/2), 1.5]

# Step length and number
h = 0.01
N = 1500
t = np.linspace(0, N*h, N)

# Solve the differential equation
y_lst = []
for v0 in v0_lst:

	# with PEFRL
	y = de.pefrl(f2, [1, 0], [0, v0], t)
	y_lst.append(y)

# Plotting
plt.title(r'Orbit for different initial velocities $v_0$')
plt.xlabel(r'$x$ coordinate')
plt.ylabel(r'$y$ coordinate')
plt.xlim(-3, 1.5)
plt.ylim(-2, 4)
plt.grid()

color_cycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd']

plt.plot(0, 0, 'ro')

for i in range(len(v0_lst)):
	plt.plot(y_lst[i][:,0], y_lst[i][:,1], color=color_cycle[i], label=fr'$v_0={v0_lst[i]:.4f}$')
	if v0_lst[i] >= 2**(1/2):
		plt.plot(y_lst[i][:,0], -y_lst[i][:,1], '--', color=color_cycle[i])

plt.legend()
plt.savefig('images/different_v0.png')
plt.show()
"""

"""
# List of initial velocities
v0min = 0.1
v0max = 1.3
v0num = 200
v0_lst = np.linspace(v0min, v0max, v0num)

# Step size and number
h = 0.001  # 0.001
N = 10000
t = np.linspace(0, N*h, N)

# Calsulate periods and distance miss
periods = []
ret_xs = []
for v0 in v0_lst:
	y = de.pefrl(f2, [1, 0], [0, v0], t)
	
	try:
		periods.append(pers(y, t)[-1][1])
	except IndexError:
		periods.append(np.nan)

	try:
		ret_xs.append(rtrnpts(y)[-1][1])
	except IndexError:
		ret_xs.append(np.nan)

# Plotting
fig, ax = plt.subplots()

ax.set_title(fr'Period $T$ as a Function of Initial Velocity $v_0$, $h={h}$, $N={N}$')
ax.set_xlabel(r'Initial Velocity $v_0$')
ax.set_ylabel(r'Period $T$')
ax.grid()

ax.plot(v0_lst, periods, label='Calculated')
ax.plot(v0_lst, T_exact(v0_lst), 'k-', label='Analytical')

axin = fig.add_axes([0.20, 0.37, 0.40, 0.40])
axin.set_title('Relative Error, log scale')
axin.set_yscale('log')
axin.grid()

axin.plot(v0_lst, np.abs(periods/T_exact(v0_lst) - 1))

ax.legend(loc='upper right')
plt.savefig('images/period_v0.png')
plt.show()


# Plotting
fig, ax = plt.subplots()

ax.set_title(fr'Return Coordinate $x$ as a Function of Initial Velocity $v_0$, $h={h}$, $N={N}$')
ax.set_xlabel(r'Initial Velocity $v_0$')
ax.set_ylabel(r'Return Coordinate $x$')
ax.set_ylim(-0.1, 1.55)
ax.grid()

ax.plot(v0_lst, ret_xs, label='Calculated')
ax.axhline(y=1, color='k', linestyle='-', label='Analytical')

axin = fig.add_axes([0.45, 0.17, 0.40, 0.40])
axin.set_title('Relative Error, log scale')
axin.set_yscale('log')
axin.grid()

axin.plot(v0_lst, np.abs(np.array(ret_xs) - 1))

ax.legend(loc='upper right')
plt.savefig('images/returnx_v0.png')
plt.show()
"""

"""
# Typical behavior near for small v0

# Initial velocity
v0 = 0.146

# Step size and number
h = 0.001
N = 50000
t = np.linspace(0, N*h, N)

# Calsulate trajectory
y = de.pefrl(f2, [1, 0], [0, v0], t)

# Plotting
fig, ax = plt.subplots()

ax.set_title(fr'Orbit for small initial velocity $v_0 = {v0}$')
ax.set_xlabel(r'$x$ coordinate')
ax.set_ylabel(r'$y$ coordinate')
ax.set_xlim(-0.25, 1.9)
ax.set_ylim(-0.35, 0.15)
ax.grid()

ax.plot(0, 0, 'ro')
ax.plot(y[:,0], y[:,1])

text = (fr'Periapsis:' + '\n' +
		fr'$r_p={(v0**2/(2-v0**2)):0.3g}$' + '\n' +
		fr'$v_p={((2-v0**2)/v0):0.3g}$')
boxprops = dict(boxstyle='round', facecolor='ivory', alpha=0.5)
ax.text(0.05, 0.27, text, transform=ax.transAxes, fontsize=14,
		  verticalalignment='top', bbox=boxprops)

plt.savefig('images/orbit_smallv0.png')
plt.show()
"""

"""
# Conservation for different v0
v0_lst = [0.7, 1.2, 1.5]

# Step length and number
h = 0.001
N = 15000
t = np.linspace(0, N*h, N)

# Solve the differential equation
y_lst = []
for v0 in v0_lst:

	# with PEFRL
	y = de.pefrl(f2, [1, 0], [0, v0], t)
	y_lst.append(y)

# erg max around 10**(-11)
# angmom max around 10**(-14)
# rnglnz max around ...
"""

"""
RKF Section maybe
"""