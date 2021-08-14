import bvp_lib as bvp
import diffeq as de
import scipy.integrate as itg
import scipy.signal as sg

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
rc('text', usetex=True)

def n(x):
	return np.heaviside(x - 1, 1) + (2 - x**2/2) * np.heaviside(1 - x, 0)

# Differential equation in eig_shoot form
def g(lmd):
	def f(y, t):
		return np.array([y[1], (-1/(4*t**2) - n(t)**2*k**2 + lmd**2)*y[0]])

	return f

# Differential equation in standard form
def f(t, y):
	return np.array([y[1], (-1/(4*t**2) - n(t)**2*k**2 + lmd**2)*y[0], np.zeros(len(y[1]))])

# Boundary conditions
def bc(ya, yb):
	return np.array([ya[0], yb[0]])

# Solve for solution
def solve_eig(l, x_max, lmd, x_num, tol):

	# Differential equation in standard form
	def f(t, y):
		return np.array([y[1], (-1/(4*t**2) - n(t)**2*k**2 + lmd**2)*y[0], np.zeros(len(y[1]))])

	# Boundary conditions
	def bc(ya, yb):
		return np.array([ya[0], yb[0], ya[1] - 1])

	# Create s interval
	x = np.linspace(tol, x_max, x_num)

	# Initial conditions from RK4
	y0 = de.rku4(lambda y, t:f(t, y), [0, 1, E0], x)

	# Solve BVP with solve_bvp
	sol = itg.solve_bvp(f, bc, x, y0.T, tol=tol)

	# Solve again using RK4
	#y_rk4 = de.rku4(lambda y, t:f(t, y), sol.y[:,0], s)
	#plt.plot(y_rk4[:,0], -y_rk4[:,1], label='Re:RK4')

	return sol


def numerov(k, y0, y1, h, N):
	y = [y0, y1]
	while len(y) < N:
		y.append((1) / (1))
	


# single mode fiber until 2.7614835
k = 0.805
tol = 10**(-5)
xmax = 5
xnum = 1000
x = np.linspace(tol, xmax, xnum)

y, lmd = bvp.eig_shoot(g, 0, 0, k, 2*k, x, adot=10, tol=tol, max_itr=150, bis=True)

plt.title(fr'Some Solutions, $k={k}$, $x_{{\mathrm{{max}}}}={xmax}$, $N={xnum}$')
plt.xlabel(r'Raidal Distance $x$')
plt.ylabel(r'Solution $R(x)$')
plt.grid()

plt.plot(x, y[:,0], label=fr'$0$-node, $\lambda={lmd:.3g}$')


plt.legend()
#plt.savefig('images/some_sols.png')
plt.show()


def sweep(k, k_min, k_max):
	y, lmd = bvp.eig_shoot(g, 0, 0, k_min, k_max, x, adot=10, tol=tol, max_itr=150, bis=True)

	if y is None:
		y1, lmd1 = sweep(k_min, k_max/2)
		y2, lmd2 = sweep(k_max/2, k_max)



# single mode fiber until 2.7614835
k = 8
tol = 10**(-5)
xmax = 20
xnum = 1000
x = np.linspace(tol, xmax, xnum)
#x = np.linspace(xmax, 0, xnum)
"""
y1, lmd1 = bvp.eig_shoot(g, 0, 0, 15, 16, x, adot=10, tol=tol, max_itr=150, bis=True)


plt.title(fr'Some Solutions, $k={k}$, $x_{{\mathrm{{max}}}}={xmax}$, $N={xnum}$')
plt.xlabel(r'Raidal Distance $x$')
plt.ylabel(r'Solution $R(x)$')
plt.grid()

plt.plot(x, y1[:,0], label=fr'$0$-node, $\lambda={lmd1:.3g}$')
#plt.plot(x, y2[:,0], label=fr'$1$-node, $\lambda={lmd2:.3g}$')
#plt.plot(x, y3[:,0], label=fr'$2$-node, $\lambda={lmd3:.3g}$')
#plt.plot(x, y4[:,0], label=fr'$3$-node, $\lambda={lmd4:.3g}$')

plt.legend()
#plt.savefig('images/some_sols.png')
plt.show()
"""

"""
# Finding all branches of the dispersion relation
k_lst = np.linspace(0.8, 10, 100)
lmd_lst1 = []
lmd_lst2 = []
lmd_lst3 = []
lmd_lst4 = []
lmd_lst5 = []
for k in k_lst:

	# first mode
	y, lmd = bvp.eig_shoot(g, 0, 0, max(k, 2.0006*k-1.2), 2.063*k - 0.5, x, adot=10, tol=tol, lmd_tol=tol, max_itr=150, bis=True)
	lmd_lst1.append(lmd)

	if k > 2.7614835:
		y, lmd = bvp.eig_shoot(g, 0, 0, max(k, 2.069*k-3.3), 2.069*k-2.5, x, adot=10, tol=tol, lmd_tol=tol, max_itr=150, bis=True)
		lmd_lst2.append(lmd)
	else:
		lmd_lst2.append(np.nan)
	
	if k > 4.8:
		y, lmd = bvp.eig_shoot(g, 0, 0, max(k, 2.187*k-6), 2.187*k-5.4, x, adot=10, tol=tol, lmd_tol=tol, max_itr=150, bis=True)
		lmd_lst3.append(lmd)
	else:
		lmd_lst3.append(np.nan)

	if k > 6.9:
		y, lmd = bvp.eig_shoot(g, 0, 0, max(k, 2.4279*k-10.3), 2.4279*k-9.3, x, adot=10, tol=tol, lmd_tol=tol, max_itr=150, bis=True)
		lmd_lst4.append(lmd)
	else:
		lmd_lst4.append(np.nan)

	if k > 8.8:
		y, lmd = bvp.eig_shoot(g, 0, 0, k, 2.4279*k-10.2, x, adot=10, tol=tol, lmd_tol=tol, max_itr=150, bis=True)
		lmd_lst5.append(lmd)
	else:
		lmd_lst5.append(np.nan)

plt.title(fr'Dispersion Relation, $x_{{\mathrm{{max}}}}={xmax}$, $N={xnum}$')
plt.xlabel(r'Wavenumber $k$')
plt.ylabel(r'Eigenvalue $\lambda(k)$')
plt.grid()

plt.plot(k_lst, lmd_lst1, 'o-', markersize=3, label=r'$0$-node branch')
plt.plot(k_lst, lmd_lst2, 'o-', markersize=3, label=r'$1$-node branch')
plt.plot(k_lst, lmd_lst3, 'o-', markersize=3, label=r'$2$-node branch')
plt.plot(k_lst, lmd_lst4, 'o-', markersize=3, label=r'$3$-node branch')
plt.plot(k_lst, lmd_lst5, 'o-', markersize=3, label=r'$4$-node branch')

plt.plot(np.linspace(0.75, 10.05, 150), np.linspace(0.75, 10.05, 150), color='grey', label='region of interest')
plt.plot(np.linspace(0.75, 10.05, 150), 2*np.linspace(0.75, 10.05, 150), color='grey')

plt.axvline(2.7614835, color='k', label='single mode limit')

plt.legend(loc='upper left')
plt.savefig('images/dispersion.png')
plt.show()
"""

"""
#Solving the problem with the finite difference method

# Define input function for lin_fd method
def v(t, lmd):
	return np.array((-1/(4*t**2) - n(t)**2*k**2 + lmd**2), dtype=np.float64)

# Tolerance on second end
b_tol = 10**(-14)

# Define parameters
k = 9.5
xmax = 15
xnum = 1000
x = np.linspace(b_tol, xmax, xnum)

# Interval on which to search for eigenvalues
E_num = 3000
E_lst = np.linspace(k, 2*k, E_num)


# Absolute maximal deviations from 0
y_amp_lst = []

# loop over energies
for E in E_lst:

	# indicator
	print(f'Calculating for E={E}')
	
	# solve the equation
	y = bvp.lin_fd(0, v(x, E), 0, x, 0, b_tol)

	# find the maximum
	print(sg.find_peaks(np.abs(y))[0], y[sg.find_peaks(np.abs(y))[0]])
	amp = np.abs(y[sg.find_peaks(np.abs(y))[0]])

	y_amp_lst.append(amp[0] if len(amp) != 0 else 0)

	#if -0.065 < E < -0.06:
	#	plt.plot(y)
	#	plt.show()

# Found eigenvalues
eigs = E_lst[sg.find_peaks(y_amp_lst)[0]]

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Eigenvalue $\lambda$')
plt.ylabel(r'Maximal Amplitude $|R(x)|_{\mathrm{max}}$')
plt.title(fr'Scan for Eigenvalues, Finite Difference Method, $N_E = {E_num}$, $x_{{\mathrm{{max}}}}={xmax}$')
plt.yscale('log')
plt.grid()

plt.plot(E_lst, y_amp_lst/max(y_amp_lst), label='Response')
plt.axvline(eigs[0], color='k', linestyle='--', label='Eigenvalues')

for E in eigs:
	plt.axvline(E, color='k', linestyle='--')

plt.legend(loc='best')
plt.savefig('images/sweep2_fd.png')
plt.show()

print(f'The energy eigenvalues between {E1} and {E2} are:')
print(eigs)
"""