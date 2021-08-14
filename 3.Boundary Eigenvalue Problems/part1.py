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

# Differential equation in eig_shoot form
def g(E):
	def f(y, t):
		return np.array([y[1], (-2/t + l*(l+1)/t**2 - E)*y[0]])

	return f

# Differential equation in standard form
def f(t, y):
	return np.array([y[1], (-2/t + l*(l+1)/t**2 - y[2])*y[0], np.zeros(len(y[1]))])

# Boundary conditions
def bc(ya, yb):
	return np.array([ya[0], yb[0]])

# Solve for solution
def solve_eig(l, x_max, E0, x_num, tol):

	# Differential equation in standard form
	def f(t, y):
		return np.array([y[1], (-2/t + l*(l+1)/t**2 - y[2])*y[0], 0*y[1]])

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

def analR(x, l, n):
	if n == 1 and l == 0:
		return 2 * x * np.exp(-x)
	elif n == 2 and l == 0:
		return 1/2**(1/2) * x * (1 - x/2) * np.exp(-x/2)
	elif n == 2 and l == 1:
		return 1/24**(1/2) * x**2 * np.exp(-x/2)


def numerov(k, y0, y1, x):
	y = [y0, y1]
	ks = k(x)
	h = x[-1] - x[-2]
	for i in range(2, len(x)):
		y.append((2*y[-1]*(1 - 5*h**2/12*ks[i-1]**2) - y[-2]*(1 + h**2/12*ks[i-2]**2)) / (1 + h**2/12*ks[i]**2))

	return np.array(y)


l = 0
tol = 10**(-8)
xmax = 25
xnum = 3000
x_fwd = np.linspace(tol, xmax, xnum)
x_bwd = np.linspace(xmax, tol, xnum)

"""
y0, E = bvp.eig_shoot(g, 0, 0, -1.1, -0.9, x_fwd, tol=tol, max_itr=150, bis=True)
norm0 = itg.simpson(y0[:,0], x_fwd)

for adot in [0.000001, 0.0001, 10, 1000, 100000]:
	y, E = bvp.eig_shoot(g, 0, 0, -1.1, -0.9, x_fwd, adot=adot, tol=tol, max_itr=150, bis=True)
	norm = itg.simpson(y[:,0], x_fwd)

	plt.plot(x_fwd, np.abs(y[:,0]/norm - y0[:,0]/norm0), label=fr'$\dot{{R}}(0)={adot:.2g}$')

plt.title(fr'Dependence on $\dot{{R}}(0)$, $l={l}$, $E={E}$')
plt.xlabel(r'Radial Distance $x$')
plt.ylabel(r'Wavefunction $|R(x) - R_{1}(x)|$')
plt.yscale('log')
plt.grid()

plt.axhline(tol, color='k', label='Tolerance')

plt.legend()
plt.savefig('images/adot.png')
plt.show()
"""

"""
plt.title(fr'First Three Solutions')
plt.xlabel(r'Radial Distance $x$')
plt.ylabel(r'Wavefunction $R(x)$')
plt.grid()

plt.plot(x_fwd, analR(x_fwd, 0, 1), label=r'$n=1$, $l=0$')
plt.plot(x_fwd, analR(x_fwd, 0, 2), label=r'$n=2$, $l=0$')
plt.plot(x_fwd, analR(x_fwd, 1, 2), label=r'$n=2$, $l=1$')

plt.legend()
plt.savefig('images/sols1.png')
plt.show()
"""

"""
E_ints = [[-1.1, -0.9], [-0.27, -0.24], [-0.27, -0.24]]

for i in range(len(E_ints)):
	Es = E_ints[i]

	if i == 2:
		l = 1
	else:
		l = 0
	
	y, E = bvp.eig_shoot(g, 0, 0, Es[0], Es[1], x_bwd, adot=10, tol=tol, max_itr=150, bis=True)
	norm = (itg.simpson(y[:,0]**2, x_fwd))**(1/2)

	n = (1 if i == 0 else 2)

	plt.plot(x_bwd, np.abs(y[:,0]/(norm*np.sign(y[-1,1])) - analR(x_bwd, l, n)), label=fr'$n={n}$, $l={l}$')

plt.title(fr'Numerical error in $R(x)$, $\epsilon={tol}$, $x_{{\mathrm{{max}}}}={xmax}$, $N={xnum}$')
plt.xlabel(r'Radial Distance $x$')
plt.ylabel(r'Wavefunction Error $|R(x) - R_{\mathrm{exact}}(x)|$')
plt.yscale('log')
plt.grid()

plt.axhline(tol, color='k', label='Tolerance')

plt.legend()
plt.savefig('images/error_bwd.png')
plt.show()
"""

"""
E_ints = [[-1.1, -0.9], [-0.27, -0.24], [-0.27, -0.24]]
xmax_lst = np.linspace(5, 30, 30)
y_err = [[], [], []]
erg_err = [[], [], []]

for xmax in xmax_lst:
	print(xmax)
	x_fwd = np.linspace(tol, xmax, xnum)
	
	for i in range(len(E_ints)):
		Es = E_ints[i]

		if i == 2:
			l = 1
		else:
			l = 0
	
		y, E = bvp.eig_shoot(g, 0, 0, Es[0], Es[1], x_fwd, adot=10, tol=tol, max_itr=150, bis=True)
		norm = (itg.simpson(y[:,0]**2, x_fwd))**(1/2)

		n = (1 if i == 0 else 2)

		y_err[i].append(np.max(np.abs(y[:,0]/norm - analR(x_fwd, l, n))))

		erg_err[i].append(np.abs(E+1/n**2))

plt.title(fr'Numerical Error in $R(x)$, $\epsilon={tol}$, $N={xnum}$')
plt.xlabel(r'Maximum Distance $x_{\mathrm{max}}$')
plt.ylabel(r'Wavefunction Error $|R(x) - R_{\mathrm{exact}}(x)|_{\mathrm{max}}$')
plt.yscale('log')
plt.grid()

labels = [r'$n=1$, $l=0$', r'$n=2$, $l=0$', r'$n=2$, $l=1$']
for i in [0, 1, 2]:
	plt.plot(xmax_lst, y_err[i], label=labels[i])

plt.legend()
plt.savefig('images/error_y_xmax.png')
plt.show()

plt.title(fr'Numerical Error in Energy, $\epsilon={tol}$, $N={xnum}$')
plt.xlabel(r'Maximum Distance $x_{\mathrm{max}}$')
plt.ylabel(r'Energy Error $|E - E_{\mathrm{exact}}|$')
plt.yscale('log')
plt.grid()

labels = [r'$n=1$, $l=0$', r'$n=2$, $l=0$', r'$n=2$, $l=1$']
for i in [0, 1, 2]:
	plt.plot(xmax_lst, erg_err[i], label=labels[i])

plt.legend()
plt.savefig('images/error_erg_xmax.png')
plt.show()
"""

"""
E_ints = [[-1.001, -0.999], [-0.251, -0.2499], [-0.251, -0.2499]]
h_lst = np.linspace(0.0001, 1.5, 40)
erg_err = [[], [], []]

for h in h_lst:
	print(h)
	x_fwd = np.linspace(tol, xmax, int(xmax/h))
	
	for i in range(len(E_ints)):
		Es = E_ints[i]

		if i == 2:
			l = 1
		else:
			l = 0
	
		_, E = bvp.eig_shoot(g, 0, 0, Es[0], Es[1], x_fwd, adot=10, tol=tol, max_itr=150, bis=True)
		print(E)
		
		n = (1 if i == 0 else 2)
		erg_err[i].append(np.abs(E+1/n**2))

plt.title(fr'Numerical Error in Energy, $\epsilon={tol}$, $x_{{\mathrm{{max}}}}={xmax}$')
plt.xlabel(r'Step Size $h$')
plt.ylabel(r'Energy Error $|E - E_{\mathrm{exact}}|$')
plt.xscale('log')
plt.yscale('log')
plt.grid()

labels = [r'$n=1$, $l=0$', r'$n=2$, $l=0$', r'$n=2$, $l=1$']
for i in [0, 1, 2]:
	plt.plot(h_lst, erg_err[i], label=labels[i])

plt.legend()
plt.savefig('images/error_erg_h_alt2.png')
plt.show()
"""



#sol = solve_eig(l, xmax, -0.1, xnum, tol)


#plt.plot(sol.y[0], label=f'{sol.y[2][-1]}')
#plt.plot(x_bwd, -y[:,0]/norm, label=f'{E}')
#plt.show()
 

"""
#Solving the problem with the finite difference method

# Define input function for lin_fd method
def v(t, E):
	return np.array((-2/t + l*(l+1)/t**2 - E), dtype=np.float64)


# Interval on which to search for eigenvalues
E_num = 3000
E_lst = -1 * np.linspace(0.9, 9, E_num)**(-2)

# Tolerance on second end
b_tol = 10**(-14)

# Define parameters
l = 0
xmax = 70
xnum = 10000
x = np.linspace(b_tol, xmax, xnum)

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

plt.xlabel(r'Energy $E$')
plt.ylabel(r'Maximal Amplitude $|R(x)|_{\mathrm{max}}$')
plt.title(fr'Scan for Eigenvalues, Finite Difference Method, $N_E = {E_num}$, $x_{{\mathrm{{max}}}}={xmax}$')
plt.yscale('log')
plt.grid()

plt.plot(E_lst, y_amp_lst/max(y_amp_lst), label='Response')
plt.axvline(eigs[0], color='k', linestyle='--', label='Eigenvalues')

for E in eigs:
	plt.axvline(E, color='k', linestyle='--')

plt.legend(loc='best')
plt.savefig('images/sweep1_fd.png')
plt.show()

print(f'The energy eigenvalues between {E1} and {E2} are:')
print(eigs)
"""

"""
xmax_lst = np.linspace(5, 70, 70)
l_lst = [0, 1, 2]
n_lst = [[], [], []]

for l in l_lst:
	for xmax in xmax_lst:
		print(f'xmax={xmax}')
		x_fwd = np.linspace(tol, xmax, xnum)
	
		n = l + 1
		y, E = bvp.eig_shoot(g, 0, 0, -1/n**2*(1 - 10**(-2)), -1/n**2*(1 + 10**(-2)), x_fwd, adot=10, tol=tol, max_itr=150, bis=True)
	
		while y is not None:
			n += 1
			y, E = bvp.eig_shoot(g, 0, 0, -1/n**2*(1 - 10**(-5)), -1/n**2*(1 + 10**(-5)), x_fwd, adot=10, tol=tol, max_itr=150, bis=True)

		n_lst[l].append(n - l - 1)	

plt.title(fr'Number of Eigenvalues, $\epsilon={tol}$, $N={xnum}$')
plt.xlabel(r'Maximum Distance $x_{\mathrm{max}}$')
plt.ylabel(r'Number of Eigenvalues $\tilde{n}$')
plt.grid()

for l in l_lst:
	plt.plot(xmax_lst, n_lst[l], label=fr'$l={l}$')

plt.legend()
plt.savefig('images/xmax_n.png')
plt.show()
"""