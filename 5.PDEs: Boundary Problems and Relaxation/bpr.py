#import bvp_lib as bvp
#import diffeq_2 as de
import scipy.integrate as itg
import scipy.signal as sg

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt

from memory_profiler import profile
from timeit import default_timer as timer
import search


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
rc('text', usetex=True)

labels = {'Jac': 'Jacobi', 'GS': 'Gauss-Seidel', 'SOR': r'SOR($\omega_0$)', 'Cheb': r'Chebyshev($\omega_0$)'}
areas = {'square': 1, 'circle': np.pi/4, 'halfcircle': np.pi/8, 'birdhouse': 5/9, 'astroid': 3 * np.pi / 32, 'circlebar': np.pi/4}

def draw_profile(x, y, u, N, tol, shape, levels=100, savefig=False):
	plt.title(fr'Flow Profile, $\epsilon={tol}$, $N={N}$')
	plt.xlabel(r'$x$ coordinate')
	plt.ylabel(r'$y$ coordinate')

	X, Y = np.meshgrid(x,y)
	plt.contour(X, Y, u, levels/10, colors='k', linestyles='dashed', linewidths=1)
	plt.contourf(X, Y, u, cmap='jet', levels=levels)
	
	clb = plt.colorbar()
	clb.ax.set_title(r'Velocity $u$')

	if savefig: plt.savefig(f'images/profile_{shape}_alt.png')

def bisect(f, a, b, h, tol=0.0001):
	u = f(a)
	
	while np.abs(a - b) > tol * h:
		m = (a + b) / 2
		v = f(m)

		if (u or v) and not (u and v):
			b = m
		else:
			a = m

	return (a + b) / 2 



def solve(x, y, q, u0, shape, method='GS', w=None, tol=10**(-5), max_itr=700, smartdD=False, diffs=False, itrs=False):

	if (method == 'SOR' or method == 'Cheb') and w is None:
		print('You must choose a value of w when using SOR or Cheb.')
		raise Exception
	
	if diffs:
		difs = []

	def diff(u, v):
		n, m = u.shape
		return np.sum((u - v)**2) / (n * m)

	print(f"You're using the method {method}.")

	h = x[1] - x[0]
	#k = y[1] - y[0]
	#d = 2 * (1/h**2 + 1/k**2)

	u = np.array(u0)
	n, m = u.shape
	if method == 'Jac': up = np.zeros((n, m))

	dif = float('inf')
	itr = 0
	while dif > tol and itr < max_itr:
		print(f'Iteration {itr}')

		if method != 'Jac': up = np.array(u)
		
		if method != 'Cheb':
			for i in range(n):
				for j in range(m):
					if i != 0 and i != n - 1 and j != 0 and j != m - 1 and mask(x[j], y[i], shape):

						if smartdD:
							alpha = np.array([1, 1, 1, 1])
							neigh = [(j-1,i), (j,i+1), (j+1,i), (j,i-1)]
							for a in range(4):
								p = neigh[a]
								if not mask(x[p[0]], y[p[1]], shape):
									if a % 2 == 0:
										f = lambda x: mask(x, y[i], shape)
										calc = np.abs(bisect(f, x[j], x[p[0]], h) - x[j]) / h
										alpha[a] = calc if calc > 0.001 else 1
									else:
										f = lambda y: mask(x[j], y, shape)
										calc = np.abs(bisect(f, y[i], y[p[1]], h) - y[i]) / h
										alpha[a] = calc if calc > 0.001 else 1
							
							if np.all(alpha != [1, 1, 1, 1]):
								print(alpha)
								c = np.floor(alpha)

								trm = 0
								for a in range(4):
									b = (a + 2) % 4
									p = neigh[a]

									trm += 2 * c[a]/(alpha[a] * (alpha[a] + alpha[b])) * u[p[0],p[1]]

								u[i,j] = np.product(alpha)/(2*(alpha[0]*alpha[2] + alpha[1]*alpha[3])) * (-q[i,j]*h**2 + trm)

						if method == 'Jac':
							if itr % 2 == 0:
								#up[i,j] = (-q[i,j] + (u[i+1,j] + u[i-1,j])/h**2 + (u[i,j+1] + u[i,j-1])/k**2)/d
								up[i,j] = (-q[i,j] * h**2 + u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])/4
							elif itr % 2 == 1:
								#u[i,j] = (-q[i,j] + (up[i+1,j] + up[i-1,j])/h**2 + (up[i,j+1] + up[i,j-1])/k**2)/d
								u[i,j] = (-q[i,j] * h**2 + up[i+1,j] + up[i-1,j] + up[i,j+1] + up[i,j-1])/4
						elif method == 'GS':
							#u[i,j] = (-q[i,j] + (u[i+1,j] + u[i-1,j])/h**2 + (u[i,j+1] + u[i,j-1])/k**2)/d
							u[i,j] = (-q[i,j] * h**2 + u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])/4
						elif method == 'SOR':
							#u[i,j] = (-q[i,j] + (u[i+1,j] + u[i-1,j])/h**2 + (u[i,j+1] + u[i,j-1])/k**2)/d
							u[i,j] = (1 - w) * u[i,j] + w * ((-q[i,j] * h**2 + u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])/4)
		
		if method == 'Cheb':
			for cl in [0, 1]:
				for i in range(n):
					for j in range(m):
						if i != 0 and i != n - 1 and j != 0 and j != m - 1 and (i + j) % 2 == cl and mask(x[j], y[i], shape):
							#u[i,j] = (-q[i,j] + (u[i+1,j] + u[i-1,j])/h**2 + (u[i,j+1] + u[i,j-1])/k**2)/d
							u[i,j] = (1 - w) * u[i,j] + w * ((-q[i,j] * h**2 + u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])/4)

		dif = diff(u, up)
		if diffs: difs.append(dif)
		itr += 1

	if dif > tol:
		print('Iteration limit exceded.')
		print('Your solution may not be up to tolerance.')
		print('dif = ', dif)

	if diffs:
		if method == 'Jac':
			if itr % 2 == 0: return u, difs
			elif itr % 2 == 1: return up, difs
		else:
			return u, difs
	elif itrs:
		if method == 'Jac':
			if itr % 2 == 0: return u, itr
			elif itr % 2 == 1: return up, itr
		else:
			return u, itr
	else:
		if method == 'Jac':
			if itr % 2 == 0: return u
			elif itr % 2 == 1: return up
		else:
			return u

def mask(x, y, shape):

	if shape == 'square':
		return True
	
	elif shape == 'circle':
		cond1 = ((x - 1/2)**2 + (y - 1/2)**2 <= 1/4)

		return cond1

	elif shape == 'circlebar':
		cond1 = ((x - 1/2)**2 + (y - 1/2)**2 <= 1/4)
		cond2 = x - 1/2 > 0 and np.abs(y - 1/2) < 0.05

		return cond1 and (not cond2)

	elif shape == 'halfcircle':
		cond1 = ((x - 1/2)**2 + (y - 1/2)**2 <= 1/4)
		cond2 = y >= 1/2

		return cond1 and cond2
	
	elif shape == 'birdhouse':
		cond1 = (x < 1/3 and y < 1/3)
		cond2 = (x > 2/3 and y < 1/3)
		cond3 = (y > x + 2/3)
		cond4 = (y > -x + 4/3 and y > 2/3)

		return not (cond1 or cond2 or cond3 or cond4)
	elif shape == 'astroid':
		cond1 = (np.abs(x - 1/2)**(2/3) + np.abs(y - 1/2)**(2/3) <= (1/2)**(2/3))

		return cond1
	else:
		print(f'The given shape "{shape}" is not implemented.')
		raise Exception


def C(u, x, y, shape):
	S = areas[shape]
	Phi = itg.simpson(itg.simpson(u, x), y)
	
	return 8 * np.pi * Phi / S**2

def find_optimal(x, y, q, u0, shape, method, m_tol=10**(-8), w_a=None, w_pm=0.1,  w_tol=10**(-3)):

	if w_a == None:
		N = len(x)
		w_a = 2 / (1 + np.pi / N)

	def f(w):
		u, itr = solve(x, x, q, u0, shape=shape, method=method, w=w, tol=m_tol, itrs=True)
		return itr

	a, b = search.gss(f, w_a - w_pm, w_a + w_pm, tol=w_tol)

	return (b + a) / 2


"""
# Finding optimal w for SOR and Cheb, take 1
N = 100
tol = 10**(-7)
shape = 'square'
x = np.linspace(0, 1, N)
u0 = np.zeros((N, N))
q = -np.ones((N, N))


w_lst = np.linspace(1.90, 1.94, 50)
itrs_sor = []
itrs_cheb = []
for w in w_lst:

	print(f'Calculating for w={w}')

	u, difs = solve(x, x, q, u0, shape=shape, method='SOR', w=w, tol=tol, diffs=True)
	itrs_sor.append(len(difs))

	u, difs = solve(x, x, q, u0, shape=shape, method='Cheb', w=w, tol=tol, diffs=True)
	itrs_cheb.append(len(difs))
	
# Plotting
figure, axes = plt.subplots(1)

plt.title(fr'Number of Iterations for Different $\omega$, $\epsilon={tol}$, $N={N}$')
plt.xlabel(r'Parameter $\omega$')
plt.ylabel(r'Number of Iterations $n$')
plt.grid()

plt.axhline(500, color='k', label='Iteration Cutoff')
plt.plot(w_lst, itrs_sor, label=r'SOR')
plt.plot(w_lst, itrs_cheb, label=r'Chebyshev')

plt.legend(loc='best')
plt.savefig('images/iters_w_square.png')
plt.show()
"""

"""
# Finding optimal w for SOR and Cheb, take 2
N_step = 3
N_max = 100
N_lst = range(2, N_max, N_step)
tol = 10**(-10)
shape = 'birdhouse'

w0_sor_lst = []
w0_cheb_lst = []
for N in N_lst:

	print(f'Calculating for N={N}')

	x = np.linspace(0, 1, N)
	u0 = np.zeros((N, N))
	q = -np.ones((N, N))

	w0 = find_optimal(x, x, q, u0, shape, 'SOR', m_tol=tol, w_pm=0.05)
	w0_sor_lst.append(w0)

	w0 = find_optimal(x, x, q, u0, shape, 'Cheb', m_tol=tol, w_pm=0.05)
	w0_cheb_lst.append(w0)

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Number of Points $N$')
plt.ylabel(r'Optimal Parameter $\omega_0$')
plt.title(fr'Optimal Parameter for Different N, $\epsilon={tol}$, {shape}')
plt.grid()

plt.plot(np.linspace(2, N_max, 1000), 2 / (1 + np.sin(np.pi / np.linspace(2, N_max, 1000))), '-k', label='Analytic Guess')
plt.plot(N_lst, w0_sor_lst, 'o-', markersize=0.8, label=r'SOR')
plt.plot(N_lst, w0_cheb_lst, 'o-', markersize=0.8, label=r'Chebyshev')

plt.legend(loc='best')
plt.savefig(f'images/optimal_N_{shape}.png')
plt.show()

print(w0_sor_lst)
print(w0_cheb_lst)
"""


N = 200
tol = 10**(-10)
shape = 'birdhouse'
x = np.linspace(0, 1, N)
u0 = np.zeros((N, N))
q = -np.ones((N, N))

#w0 = find_optimal(x, x, q, u0, shape, 'Cheb', m_tol=tol, w_pm=0.05)
#print(w0)

u = solve(x, x, q, u0, shape=shape, method='Cheb', w=1.9536498398997866, tol=tol)

draw_profile(x, x, u, N, tol, shape)
plt.show()

print('C=', C(u, x, x, shape))





#u = solve(x, x, q, u0, shape=shape, method='SOR', w=1.939081, tol=tol)

#draw_profile(x, x, u, N, tol, shape)
#plt.show()


"""
N = 100
tol = 10**(-10)
shape = 'square'
x = np.linspace(0, 1, N)
u0 = np.zeros((N, N))
q = -np.ones((N, N))

for m in ['Jac', 'GS', 'SOR', 'Cheb']:
	u, difs = solve(x, x, q, u0, shape=shape, method=m, w=1.915 if m == 'SOR' else 1.922, tol=tol, diffs=True)
	plt.plot(difs, label=labels[m])


plt.xlabel(r'Iteration number $n$')
plt.ylabel(r'Last Difference $|w_{-1} - w_{-2}| / N^2$')
plt.title(fr'Convergence of Methods, $\epsilon={tol}$, {shape}')
plt.yscale('log')
plt.grid()

plt.legend(loc='best')
plt.savefig('images/conv.png')
plt.show()

print(C(u, x, x, shape))

draw_profile(x, x, u, N, tol, shape)
plt.show()
"""

"""
u, difs = solve(x, x, q, u0, shape=shape, method='Cheb', w=1.939081, tol=tol, diffs=True)

draw_profile(x, x, u, N, tol, shape)
plt.show()

"""





"""
#------------------------------------------------------------------------------
# Testing speed of implementation

# different N
N_num = 90
N_lst = range(2, N_num)

# Set up parameters
tol = 10**(-9)
shape = 'square'

# Initizalize list of times
times_N = {}

for N in N_lst:

	print(f'Calculating for N={N}')

	x = np.linspace(0, 1, N)
	u0 = np.zeros((N, N))
	q = -np.ones((N, N))

	for m in ['Jac', 'GS', 'SOR', 'Cheb']:
		start = timer()
		u = solve(x, x, q, u0, shape=shape, method=m, w=1.915 if m == 'SOR' else 1.922, tol=tol)
		end = timer()
		if m in times_N.keys():
			times_N[m].append(end - start)
		else:
			times_N[m] = []
			times_N[m].append(end - start)

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Number of Points $N$')
plt.ylabel(r'Evaluation Time $t$ [s]')
plt.title(fr'Evaluation Time for Different N, $\epsilon={tol}$, {shape}')
plt.xscale('log')
plt.yscale('log')
plt.grid()

plt.plot(N_lst, times_N['Jac'], 'o-', markersize=0.8, label=r'Jacobi')
plt.plot(N_lst, times_N['GS'], 'o-', markersize=0.8, label=r'Gauss-Seidel')
plt.plot(N_lst, times_N['SOR'], 'o-', markersize=0.8, label=r'SOR($\omega_0$)')
plt.plot(N_lst, times_N['Cheb'], 'o-', markersize=0.8, label=r'Chebyshev($\omega_0$)')

print(times_N)

plt.legend(loc='best')
plt.savefig('images/times_alt.png')
plt.show()
"""