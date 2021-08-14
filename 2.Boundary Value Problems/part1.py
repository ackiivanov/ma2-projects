import diffeq_2 as de
import scipy.integrate as itg

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
rc('text', usetex=True)

# Differential equation in standard form
def f(t, y):
	return np.array([np.cos(y[3]), np.sin(y[3]),
		- beta * y[0] * np.cos(y[3]) + np.sin(y[3]),
		(beta * y[0] * np.sin(y[3]) + np.cos(y[3]))/y[2]])

# Residuals for boundary conditions
def bc(ya, yb):
	return np.array([ya[0], ya[1], yb[0], yb[1] - yF])

def length(y):
	diff = np.diff(y)
	return np.sum((diff[0]**2 + diff[1]**2)**(1/2))

def solve_string(beta, yF, F0, alpha0, s_num, tol):

	# Differential equation in standard form
	def f(t, y):
		return np.array([np.cos(y[3]), np.sin(y[3]),
			- beta * y[0] * np.cos(y[3]) + np.sin(y[3]),
			(beta * y[0] * np.sin(y[3]) + np.cos(y[3]))/y[2]])

	# Residuals for boundary conditions
	def bc(ya, yb):
		return np.array([ya[0], ya[1], yb[0], yb[1] - yF])

	# Create s interval
	s = np.linspace(0, 1, s_num)

	# Initial conditions from RK4
	y0 = de.rku4(lambda y, t:f(t, y), [0, 0, F0, alpha0], s)

	# Solve BVP with solve_bvp
	sol = itg.solve_bvp(f, bc, s, y0.T, tol=tol)

	# Solve again using RK4
	#y_rk4 = de.rku4(lambda y, t:f(t, y), sol.y[:,0], s)
	#plt.plot(y_rk4[:,0], -y_rk4[:,1], label='Re:RK4')

	return sol

# Number of interval points
s_num = 1000
	
# Set angular velocity and connection point
beta_lst = np.linspace(1, 20, 200)
yF = 0.9


xmax_lst = []
ymin_lst = []
lengths = []
for beta in beta_lst:

	# Set initial condtitions
	F0 = beta/15
	alpha0 = 0

	sol = solve_string(beta, yF, F0, alpha0, s_num, 10**(-8))

	L = length(sol.y)
	print('beta:', beta, '    ', 'Length:', L)

	if np.abs(sol.y[0,-1]) <= 10**(-3) and np.abs(L - 1) <= 5*10**(-2):
		lengths.append(L)
		xmax_lst.append(np.max(np.abs(sol.y[0])))
		ymin_lst.append(np.abs(np.min(sol.y[1])))
	else:
		lengths.append(np.nan)
		xmax_lst.append(np.nan)
		ymin_lst.append(np.nan)

	#if 1.9 <= beta <= 2.2:
		#plt.plot(sol.y[0], sol.y[1])
		#plt.show()

		#plt.plot(sol.y[2])
		#plt.show()

		#plt.plot(sol.y[3])
		#plt.show()

lengths = np.array(lengths)

fig, ax = plt.subplots()

ax.set_title(fr'Maximal Extension, $y_F={yF}$')
ax.set_xlabel(r'Angular Velocity $\beta$')
ax.set_ylabel(r'Extension $|y_{\mathrm{min}}|$ / Extension $x_{\mathrm{max}}$')
#ax.set_ylim(-0.01, 0.188)
ax.grid()

ax.axhline((1-yF)/2, color='firebrick', label=r'$y$ range')
ax.axhline(0, color='firebrick')

ax.plot(beta_lst, xmax_lst, label=r'$x_{\mathrm{max}}$')
ax.plot(beta_lst, ymin_lst, label=r'$|y_{\mathrm{min}}|$')

axin = fig.add_axes([0.5, 0.41, 0.37, 0.3])
axin.set_title(r'Length Deviation, log scale')
axin.set_yscale('log')
axin.grid()

axin.plot(beta_lst, np.abs(lengths - 1), color='forestgreen')

ax.legend(loc='best')
#plt.savefig('images/max_dev.png')
plt.show()




"""
# Set angular velocity and connection point
#beta_lst = [2, 3.5, 5, 10, 15] #15
#beta_lst = [20, 40, 70, 100] #35
#beta_lst = [60, 70, 100, 200] #78
beta_lst = [100, 200, 300, 400] #160
yF = 0.9
N = 160

# Solve different beta
sols = []
for beta in beta_lst:

	# Set initial condtitions
	F0 = beta/N
	alpha0 = 0

	sol = solve_string(beta, yF, F0, alpha0, s_num, 10**(-8))

	L = length(sol.y)
	print('beta:', beta, '    ', 'Length:', L)

	sols.append(sol)

plt.title(fr'String Shape for $y_F={yF}$')
plt.xlabel(r'$x$ coordinate')
plt.ylabel(r'$y$ coordinate')
plt.grid()

plt.axhline(0, color='k')
plt.axvline(0, color='k')

for i in range(len(beta_lst)):
	plt.plot(sols[i].y[0], sols[i].y[1], label=fr'$\beta = {beta_lst[i]}$')

plt.legend()
#plt.savefig(f'images/few_curves_{yF}_{N}.png')
plt.show()


plt.title(fr'Tension Force for four Extrema, $y_F={yF}$')
plt.xlabel(r'Length along the String $s$')
plt.ylabel(r'Ratio $\frac{F}{\beta}$')
plt.grid()

for i in range(len(beta_lst)):
	plt.plot(np.linspace(0, 1, s_num), sols[i].y[2]/beta_lst[i], label=fr'$\beta={beta_lst[i]}$')

plt.legend(loc='best')
plt.savefig(f'images/force_{N}')
plt.show()
"""