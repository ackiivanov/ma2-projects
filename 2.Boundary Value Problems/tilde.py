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
		- y[0] * np.cos(y[3]) + np.sin(y[3]),
		(y[0] * np.sin(y[3]) + np.cos(y[3]))/y[2]])

# Residuals for boundary conditions
def bc(ya, yb):
	return np.array([ya[0], ya[1], yb[0], yb[1] - beta * yF])
	
def length(y):
	diff = np.diff(y)
	return np.sum((diff[0]**2 + diff[1]**2)**(1/2))

def solve_string(beta, yF, F0, alpha0, s_num, tol):

	# Differential equation in standard form
	def f(t, y):
		return np.array([np.cos(y[3]), np.sin(y[3]),
			- y[0] * np.cos(y[3]) + np.sin(y[3]),
			(y[0] * np.sin(y[3]) + np.cos(y[3]))/y[2]])

	# Residuals for boundary conditions
	def bc(ya, yb):
		return np.array([ya[0], ya[1], yb[0], yb[1] - beta * yF])

	# Create s interval
	s = np.linspace(0, beta, s_num)

	# Initial conditions from RK4
	y0 = de.rku4(lambda y, t:f(t, y), [0, 0, F0, alpha0], s)

	#plt.plot(y0[:,0], -y0[:,1], label='RK4')

	# Solve BVP with solve_bvp
	sol = itg.solve_bvp(f, bc, s, y0.T, tol=tol)

	# Solve again using RK4
	#y_rk4 = de.rku4(lambda y, t:f(t, y), sol.y[:,0], s)

	#plt.plot(y_rk4[:,0], -y_rk4[:,1], label='Re:RK4')

	return sol

# Number of interval points
s_num = 1000
	
# Set angular velocity and connection point
beta = 10
yF = 0.9

# Set initial condtitions
F0 = beta**2/10
alpha0 = 0

sol = solve_string(beta, yF, F0, alpha0, s_num, 10**(-8))

L = length(sol.y)
print('beta:', beta, '    ', 'Length:', L/beta)


plt.plot(sol.y[0], -sol.y[1], label='BVP')
plt.grid()
plt.legend()
plt.show()
"""
plt.plot(sol.y[3])
plt.axhline(0, color='k')
plt.axhline(np.pi/2, color='k')
plt.show()

plt.plot(sol.y[2])
plt.show()

"""

"""
xmax_lst = []
ymax_lst = []
for beta in beta_lst:

	# Set initial condtitions
	F0 = beta/10
	alpha0 = 0.5

	sol = solve_string(beta, yF, F0, alpha0, s_num, 10**(-8))

	L = length(sol.y)
	print('beta:', beta, '    ', 'Length:', L)

	if np.abs(L - 1) <= 5*10**(-2):
		xmax_lst.append(np.abs(np.max(sol.y[0])))
		ymax_lst.append(np.max(sol.y[1]))
	else:
		xmax_lst.append(np.nan)
		ymax_lst.append(np.nan)

	if 2 <= beta <= 2.1:
		plt.plot(sol.y[0], -sol.y[1])
		plt.show()

plt.axhline((1+yF)/2, color='k')
plt.axhline(yF, color='k')


plt.plot(beta_lst, xmax_lst, label='xmax')
plt.plot(beta_lst, ymax_lst, label='ymax')
plt.legend()
plt.show()
"""
