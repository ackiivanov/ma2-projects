import bvp_lib as bvp
import diffeq_2 as de
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
def make_k(x, phi):
	return 2*Z/x + 2*phi

def make_phi(x, R):
	R2overx = R**2 / x
	nR2overx = itg.simpson(R2overx, x)
	return (-1/x * itg.cumulative_trapezoid(R**2, x, initial=0)
		+ itg.cumulative_trapezoid(R2overx, x, initial=0) - nR2overx)

def make_R0(x):
	Zstar = Z - 5/16
	return 2 * Zstar**(3/2) * x * np.exp(- Zstar * x)


def solve_2eatom(x, eps0, epspm, R1=1, max_itr=250, save_all=False, safe=True, half=False):
	if half:
		switch = 1
		cnt = 0

	R = make_R0(x)
	epsoldold = float('-inf')
	epsold = float('-inf')
	eps = eps0
	itr = 0

	if save_all:
		R_lst = []
		eps_lst = []
		phi_lst = []

	while np.abs(eps - epsold) > tol and itr < max_itr:
		print('***', np.abs(eps - epsold), np.abs(epsoldold - eps))

		if half and switch == 1:
			if np.abs(epsoldold - eps) < np.abs(epsold - eps):
				switch = 0
				cnt = 1
		elif half and switch == 0:
			cnt += 1

		if half and cnt % 2 == 1 and np.abs(epsoldold - eps) < tol:
			return R, eps, phi

		phi = make_phi(x, R)
		k = make_k(x, phi)

		Rold = R
		epsoldold = epsold
		epsold = eps

		if half and itr > 1:
			R, eps = bvp.eig_shoot(k, 0, 0, epsoldold - np.abs(epsoldold/5), epsoldold + np.abs(epsoldold/5), x,
							   a1=R1, tol=tol, max_itr=200, bis=True) #lmd_tol
		else:
			R, eps = bvp.eig_shoot(k, 0, 0, eps - epspm, eps + epspm, x,
							   a1=R1, tol=tol, max_itr=200, bis=True) #lmd_tol

		if not safe and R is None:
			return R, np.nan, phi

		nR = (np.abs(itg.simpson(R**2, x)))**(1/2)

		#plt.show()

		R = R / nR

		plt.plot(x, R)

		if save_all:
			R_lst.append(R)
			eps_lst.append(eps)
			phi_lst.append(phi)

		itr += 1

	if safe and np.abs(eps - epsold) > tol:
		print('The solution did not converge')
		return None, None, None

	if save_all:
		return R_lst, eps_lst, phi_lst

	return R, eps, phi

def erg(x, R, phi, eps):
	return 2*(eps + itg.simpson(phi*R**2, x))*13.6058


Z = 4
tol = 10**(-3)
xmax = 15
xnum = 3000
x = np.linspace(10**(-10), xmax, xnum)
#x = np.linspace(xmax, 10**(-10), xnum)
h = x[1] - x[0]

figure, axes = plt.subplots(1)

plt.plot(x, make_R0(x), label='Initial guess')

# Z=2, -1.8 0.46
# Z=3, -5.5 1.00, -2 1.50
# Z=4, -2.85 2.5
R, eps, phi = solve_2eatom(x, -2.85, 2.5, R1=10)

print('The energy is (ret):', eps)
print('The energy is (int):', erg(x, R, phi, eps))


plt.title(fr'Solution, $Z={Z}$')
plt.xlabel(r'Radial Distance $x$')
plt.ylabel(r'Radial Wavefunction $R(x)$')
plt.xlim(-0.3, 8.1)
plt.grid()

plt.plot(x, R, color='k', label='Solution')
#plt.plot(x, make_R0(x), label='Initial guess')

text = (fr'$E={erg(x, R, phi, eps):0.5g}\,\mathrm{{eV}}$')# + '\n' + r'$E_{\mathrm{real}}=-78.88\,\mathrm{eV}$')
boxprops = dict(boxstyle='round', facecolor='ivory', alpha=0.5)
axes.text(0.53, 0.6, text, transform=axes.transAxes, fontsize=14,
	verticalalignment='top', bbox=boxprops)

plt.legend()
plt.savefig(f'images/sol{Z}_second_t2.png')
plt.show()


plt.title(fr'Final Potential, $Z={Z}$')
plt.xlabel(r'Radial Distance $x$')
plt.ylabel(r'Interaction Potential $\Phi(x)$')
plt.ylim(-1, 0)
plt.xlim(-0.3, 8.1)
plt.grid()

plt.axvline(0, color='k')
plt.plot(x, phi, label='Solution')
plt.plot(x, -1/x, label=r'$-1/x$ Potential')

plt.legend()
plt.savefig(f'images/sol_phi{Z}_second.png')
plt.show()


"""
Rp_lst = [0.01, 0.1, 10, 1000, 100000]

R0, eps0, phi0 = solve_2eatom(x, -1.83, 0.5, R1=1)

Rerr = []
epserr = []
phierr = []
for Rp in Rp_lst:
	R, eps, phi = solve_2eatom(x, -1.83, 0.5, R1=Rp*h)

	Rerr.append(np.abs(R - R0))
	epserr.append(np.abs(eps - eps0))
	phierr.append(np.abs(phi - phi0))
	
plt.title(fr'Dependence on $\dot{{R}}(0)$, $Z={Z}$')
plt.xlabel(r'Radial Distance $x$')
plt.ylabel(r'Wavefunction $|R(x) - R_{1}(x)|$')
plt.yscale('log')
plt.grid()

plt.axhline(tol, color='k', label='Tolerance')
for i in range(len(Rp_lst)):
	plt.plot(x, Rerr[i], label=fr'$\dot{{R}}(0)={Rp_lst[i]}$')

plt.legend()
plt.savefig('images/R_Rp.png')
plt.show()

plt.title(fr'Dependence on $\dot{{R}}(0)$, $Z={Z}$')
plt.xlabel(r'Radial Distance $x$')
plt.ylabel(r'Interaction Potential $|\Phi(x) - \Phi_{1}(x)|$')
plt.yscale('log')
plt.grid()

plt.axhline(tol, color='k', label='Tolerance')
for i in range(len(Rp_lst)):
	plt.plot(x, phierr[i], label=fr'$\dot{{R}}(0)={Rp_lst[i]}$')

plt.legend()
plt.savefig('images/phi_Rp.png')
plt.show()

print('Energy error:', epserr)
"""

"""
xmax_lst = np.linspace(3, 10, 30)

Z_lst = [1.2, 1.7, 2, 3]
tol = 10**(-5)
xnum = 3000

ergerr = [[], [], [], [], []]
for i in range(len(Z_lst)):
	Z = Z_lst[i]

	for xmax in xmax_lst:
		print(xmax)
		x = np.linspace(10**(-10), xmax, xnum)
		
		if Z == 1.2:
			R, eps, phi = solve_2eatom(x, -0.26, 0.18, R1=10, safe=False)
		elif Z == 1.7:
			R, eps, phi = solve_2eatom(x, -1.10, 0.2, R1=10, safe=False)
		elif Z == 2:
			R, eps, phi = solve_2eatom(x, -1.8, 0.46, R1=10, safe=False)
		elif Z == 3:
			R, eps, phi = solve_2eatom(x, -5.58, 0.4, R1=10, safe=False)
		if eps is np.nan: E = np.nan
		else: E = erg(x, R, phi, eps)

		ergerr[i].append(np.abs(E))

	ergerr[i] = [np.abs(x - np.abs(E)) for x in ergerr[i]][:-1]
	
plt.title(fr'Convergence Depending on $x_{{\mathrm{{max}}}}$, $N={xnum}$')
plt.xlabel(r'Maximal Distance $x_{{\mathrm{{max}}}}$')
plt.ylabel(r'Energy $|E - E_{\mathrm{final}}|$')
plt.yscale('log')
plt.grid()

plt.axhline(tol, color='k', label='Tolerance')
for i in range(len(Z_lst)):
	plt.plot(xmax_lst[:-1], ergerr[i], label=fr'$Z={Z_lst[i]}$')

plt.legend()
plt.savefig('images/ergerr_xmax_alt2.png')
plt.show()
"""

"""
Z = 1.058
tol = 10**(-4)
xmax = 15
xnum = 5000
x = np.linspace(10**(-10), xmax, xnum)

# Z=2, -1.8 0.46
R_lst, eps_lst, phi_lst = solve_2eatom(x, -0.14, 0.1, R1=10, save_all=True, safe=False)

plt.title(fr'Convergence of Hartree-Fock, $Z={Z}$')
plt.xlabel(r'Iteration Number $n$')
plt.ylabel(r'Absolute Error in Quantity $|Q - Q_{\mathrm{final}}|$')
plt.yscale('log')
plt.grid()

R_max_lst = [np.max(np.abs(R)) for R in R_lst]
phi_max_lst = [np.max(np.abs(phi)) for phi in phi_lst]
E_lst = [erg(x, R_lst[i], phi_lst[i], eps_lst[i]) for i in range(len(R_lst))]

plt.axhline(tol, color='k', label='Tolerance')
plt.plot(np.abs(R_max_lst[:-1] - R_max_lst[-1]), 'o-', label=r'Solution $R(x)$')
plt.plot(np.abs(phi_max_lst[:-1] - phi_max_lst[-1]), 'o-', label=r'Potential $\Phi(x)$')
plt.plot(np.abs(E_lst[:-1] - E_lst[-1]), 'o-', label=r'Energy $E$')

plt.legend()
plt.savefig(f'images/convergence{Z}.png')
plt.show()


figure, axes = plt.subplots(1)

plt.title(fr'Solution, $Z={Z}$')
plt.xlabel(r'Radial Distance $x$')
plt.ylabel(r'Radial Wavefunction $R(x)$')
plt.grid()

plt.plot(x, R_lst[-1], label='Solution')
plt.plot(x, make_R0(x), label='Initial guess')

text = (fr'$E\ \ \ \, ={E_lst[-1]:0.5g}\,\mathrm{{eV}}$' + '\n' +
	fr'$E_{{\mathrm{{real}}}}={-78.88}\,\mathrm{{eV}}$')
boxprops = dict(boxstyle='round', facecolor='ivory', alpha=0.5)
axes.text(0.53, 0.6, text, transform=axes.transAxes, fontsize=14,
	verticalalignment='top', bbox=boxprops)


plt.legend()
plt.savefig(f'images/sol{Z}.png')
plt.show()
"""

"""
Z = 3
R, eps, phi = solve_2eatom(x, -5.5, 1, R1=10)

print('The energy is (ret):', eps)
print('The energy is (int):', erg(x, R, phi, eps))

figure, axes = plt.subplots(1)

plt.title(fr'Solution, $Z={Z}$')
plt.xlabel(r'Radial Distance $x$')
plt.ylabel(r'Radial Wavefunction $R(x)$')
plt.grid()

plt.plot(x, R, label='Solution')
plt.plot(x, make_R0(x), label='Initial guess')

text = (fr'$E\ \ \ \, ={erg(x, R, phi, eps):0.5g}\,\mathrm{{eV}}$' + '\n' +
	fr'$E_{{\mathrm{{real}}}}={-198.04}\,\mathrm{{eV}}$')
boxprops = dict(boxstyle='round', facecolor='ivory', alpha=0.5)
axes.text(0.53, 0.6, text, transform=axes.transAxes, fontsize=14,
	verticalalignment='top', bbox=boxprops)


plt.legend()
plt.savefig(f'images/sol{Z}.png')
plt.show()
"""

"""
xmax_lst = np.linspace(5, 30, 30)
Rerr = []
ergerr = []

for xmax in xmax_lst:
	print(xmax)
	x_fwd = np.linspace(10**(-10), xmax, xnum)
	
	R, eps, phi = solve_2eatom(x, -1.8, 0.46, R1=10)
		
	Rerr[i].append(np.max(np.abs(y[:,0]/norm - analR(x_fwd, l, n))))
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
# Z = 8 is interesting

Z_lst = [1.5]
# -1.1475*Z**2 + 2.44893*Z - 2.38143, 0.46 + 0.5*(Z-2)**2

guess = [[-1.8, 0.46], [-5.5, 0.8], [-11.3, 1], [-19.08, 1], [-28.83, 1.5]]

ergs = []
for Z in Z_lst:

	R, eps, phi = solve_2eatom(x, -0.3, 0.3, R1=10)

	print('energy:', erg(x, R, phi, eps))

	if R is None:
		print(ergs)

	ergs.append(erg(x, R, phi, eps))

plt.plot(x, R)
plt.show()


print(ergs)

plt.title(fr'Calculated Energy, $\epsilon={tol}$, $N={xnum}$')
plt.xlabel(r'Atomic Number $Z$')
plt.ylabel(r'Energy $E$ [eV]')
plt.grid()

#plt.plot(Z_lst, ergs)

#plt.legend()
#plt.savefig('images/ergs_Z.png')
#plt.show()

"""

