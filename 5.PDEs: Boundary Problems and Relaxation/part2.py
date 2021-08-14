import scipy.integrate as itg
import scipy.signal as sg

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

from memory_profiler import profile
from timeit import default_timer as timer
import search
 
# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
rc('text', usetex=True)

pr = 2
T1 = 0
T2 = 10
cur = 0

def draw_profile(x, y, u, N, tol, levels=100, savefig=False):
	w, h = figaspect(1.5)
	fig, ax = plt.subplots(figsize=(w,h))

	plt.title(fr'Profile, $\epsilon={tol}$, $N={N}$')
	plt.xlabel(r'$r$ coordinate')
	plt.ylabel(r'$z$ coordinate')
	
	X, Y = np.meshgrid(x,y)
	#plt.contour(X, Y, u, levels/10, colors='k', linestyles='dashed', linewidths=1)
	plt.contourf(X, Y, u, cmap='jet', levels=levels)
	
	clb = plt.colorbar()
	clb.ax.set_title(r'Temp. $T$')

	if savefig: plt.savefig(f'images/profile_{T2}_{cur}_{pr}.png')


def SOR(x, y, q, u0, w, pr=1, tol=10**(-5), max_itr=700, smartdD=False, diffs=False, itrs=False):

	if diffs:
		difs = []

	def diff(u, v):
		n, m = u.shape
		return np.sum(np.abs(u - v)**2) / (n * m)

	h = x[1] - x[0]

	u = np.array(u0)
	n, m = u.shape

	dif = float('inf')
	itr = 0
	while dif > tol and itr < max_itr:
		print(f'Iteration {itr}')

		up = np.array(u)
		
		for i in range(n-1, -1, -1):
			for j in range(m-1, -1, -1):
				if i == 0:
					u[i,j] = u[1,j] - h * cur
				elif i == n - 1:
					if pr == 1: u[i,j] = T1
					elif pr == 2: u[i,j] = u[n-2,j]
				elif j == m - 1 and i <= n//2:
					u[i,j] = T1
				elif j == m - 1 and i > n//2:
					u[i,j] = T2
				elif j == 0:
					u[i,j] = u[i,1]
				else:
					u[i,j] = (1 - w) * u[i,j] + 2/5 * w * (1/4 * (u[i+1,j] + u[i-1,j]) + (1 + 1/(2*(tol + j)))*u[i,j+1] + (1 - 1/(2*(tol + j)))*u[i,j-1])

		dif = diff(u, up)
		if diffs: difs.append(dif)
		itr += 1

	if dif > tol:
		print('Iteration limit exceded.')
		print('Your solution may not be up to tolerance.')
		print('dif = ', dif)

	if diffs:
		return u, difs
	elif itrs:
		return u, itr
	else:
		return u

def find_optimal(x, y, q, u0, m_tol=10**(-8), w_a=None, w_pm=0.1,  w_tol=10**(-3)):

	if w_a == None:
		N = len(x)
		w_a = 2 / (1 + np.pi / N)

	def f(w):
		print(w)
		u, itr = SOR(x, x, q, u0, w=w, tol=m_tol, itrs=True)
		return itr

	a, b = search.gss(f, w_a - w_pm, w_a + w_pm, tol=w_tol)

	return (b + a) / 2

def make_u0(N):
	u0 = 15 * np.ones((N, N))

	for i in range(N):
		if i <= N//2: u0[i,-1] = T1
		else: u0[i,-1] = T2

		u0[N-1,i] = T1

	return u0


N = 150
w = 1.9764435757741872
tol = 10**(-7)
x = np.linspace(0, 1, N)
y = np.linspace(0, 2, N)
u0 = make_u0(N)
q = -np.ones((N, N))

#w0 = find_optimal(x, y, q, u0, m_tol=tol, w_pm=0.05)
#print(w0)

u = SOR(x, y, q, u0, w, pr=pr, tol=tol)

draw_profile(x, y, u, N, tol)
plt.show()