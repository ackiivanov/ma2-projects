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
def fpr(t, y):
	return np.array([K * y[2], K * y[3],
		K * (-y[0] - 2 * y[0] * y[1]),
		K * (-y[1] - y[0]**2 + y[1]**2), K])

# Differential equation in standard form
def f(t, y):
	return np.array([y[2], y[3], (-y[0] - 2 * y[0] * y[1]),
		(-y[1] - y[0]**2 + y[1]**2)])

# Residuals for boundary conditions
def bc(ya, yb):
	return np.array([ya[0], ya[1] - y0, yb[0], yb[1] - y0])

def solve_orbit(E, y0i, t, tol):

	# Differential equation in standard form
	def f(t, y):
		return np.array([y[2], y[3], (-y[0] - 2 * y[0] * y[1]),
			(-y[1] - y[0]**2 + y[1]**2)])

	# Residuals for boundary conditions
	def bc(ya, yb):
		return np.array([ya[0] - yb[0], ya[1] - yb[1], ya[2] - yb[2], ya[3] - yb[3]])

	# Calculate initial speed
	v0i = (2*E - y0i**2 + 2/3*y0i**3)**(1/2)

	# Initial conditions from RK4
	y0 = de.rku4(lambda y, t:f(t, y), [0, y0i, v0i, 0], t)

	# Solve BVP with solve_bvp
	sol = itg.solve_bvp(f, bc, t, y0.T, tol=tol)

	# Solve again using RK4
	#y_rk4 = de.rku4(lambda y, t:f(t, y), sol.y[:,0], s)
	#plt.plot(y_rk4[:,0], -y_rk4[:,1], label='Re:RK4')

	return sol


def poinc(E, y0, v0, t):

	# Differential equation in standard form
	def f(t, y):
		return np.array([y[2], y[3], (-y[0] - 2 * y[0] * y[1]),
			(-y[1] - y[0]**2 + y[1]**2)])

	def event(t, y):
		return y[0]

	u0 = (2*E - y0**2 + 2/3*y0**3 - v0**2)**(1/2)
	print(u0)

	sol = itg.solve_ivp(f, t, [0, y0, u0, v0], events=event)

	return sol

def energy(y):
	return y[2]**2/2 + y[3]**2/2 + y[0]**2/2 + y[1]**2/2 + y[0]**2 * y[1] - y[1]**3/3

def erg(y):
	return y[0]**2/2 + y[1]**2/2 + y[0]**2 * y[1] - y[1]**3/3

"""
# Choose y0 (between 0 and 1) and E (between 0 and 1/6)
E = 0.16
y0 = 0.3
v0 = 0.1

for v0 in [0, 0.1, 0.2, 0.3]:
	sol = poinc(E, y0, v0, (0, 10000))

	plt.plot(sol.y_events[0][:,1], sol.y_events[0][:,3], 'o', markersize=0.7)

plt.title(fr'Poincar\'e section for $E={E}$')
plt.xlabel(r'$y$ coordinate')
plt.ylabel(r'$\dot{y}$ coordinate')
#plt.xlim(-0.26, 0.48)
plt.grid()
#plt.legend(loc='best')
plt.savefig(f'images/poincare_{E}.png', dpi=300)
plt.show()
"""
"""
#T_lst = [51.1864, 37.6271, 56.6101, 82.3729, 81.0169]
#T_lst = [24.0678, 25.4237]#, 26.7797]
#T_lst = [25.4237]#, 37.6271]#, 38.9831]

#T_lst = np.linspace(20, 100, 60)
maxx = []
for T in T_lst:

	print('T:', T)
	# Create time points
	N = 5000
	t = np.linspace(0, T, N)


	sol = solve_orbit(E, y0, t, 10**(-8))

	Ecalc = energy(sol.y[:,-1])

	plt.plot(sol.y[0], sol.y[1], label=fr'$E={Ecalc:.5g}$')

	maxx.append(np.max(sol.y[0]))

#plt.plot([0, 3**(1/2)/2, -3**(1/2)/2, 0], [1, -0.5, -0.5, 1])

plt.title(fr'Examples of Orbits')
plt.xlabel(r'$x$ coordinate')
plt.ylabel(r'$y$ coordinate')
plt.grid()
plt.legend(loc='best')
plt.savefig(f'images/orbits_2.png')
plt.show()
"""
"""
plt.plot(T_lst, maxx)
plt.show()
#plt.plot([0, 3**(1/2)/2, -3**(1/2)/2, 0], [1, -0.5, -0.5, 1])
plt.plot(sol.y[0], sol.y[1])
plt.show()
"""

# dummy plotting variables
x_dummy = np.linspace(-0.9, 0.9, 100)
y_dummy = np.linspace(-0.6, 1.1, 100)
erg_dummy = [[erg([x, y]) for x in x_dummy] for y in y_dummy]

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'$x$ coordinate')
plt.ylabel(r'$y$ coordinate')
plt.title(fr'Potential Portrait of H\'enon-Heiles potential')
plt.grid()

plt.contour(x_dummy, y_dummy, erg_dummy, 30, colors='k', linestyles='dashed', linewidths=1)
plt.contourf(x_dummy, y_dummy, erg_dummy, 250, alpha=0.7/3, antialiased=True)
plt.contourf(x_dummy, y_dummy, erg_dummy, 260, alpha=0.7/3, antialiased=True)
plt.contourf(x_dummy, y_dummy, erg_dummy, 270, alpha=0.7/3, antialiased=True)
plt.plot([0, 3**(1/2)/2, -3**(1/2)/2, 0], [1, -0.5, -0.5, 1])

clb = plt.colorbar()
clb.ax.set_title(r'Energy $E$')
plt.savefig('images/portrait.png')
plt.show()