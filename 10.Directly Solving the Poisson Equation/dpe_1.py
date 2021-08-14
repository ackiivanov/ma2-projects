import numpy as np
import scipy.linalg as la
import scipy.fft as fft

import matplotlib.pyplot as plt
from matplotlib import rc
from FitterPlotter import FitterPlotter

from timeit import default_timer as timer
import timeit
import time

# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# boundaries of the different shapes
bounds = {'hom': [[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], 'F': [[0, 1/4, 1/4, 1/2,
	1/2, 3/4, 3/4, 1/2, 1/2, 1, 1, 1/4, 1/4, 0, 0], [1/4, 1/4, 0, 0, 1/4, 1/4,
	1/2, 1/2, 3/4, 3/4, 1, 1, 1/2, 1/2, 1/4]]}

# Create the matrix g
def make_g(N, M, dist, rho=None):

	if dist == 'hom':
		return np.ones((N, M))

	# initialize matrix g
	g = np.zeros((N, M))

	if dist == 'F':

		# set density if not given
		if rho is None:
			rho = [10, 1]

		# make sure denisty list is correct size
		assert len(rho) == 2
		
		# loop over point in domain
		for i in range(N):
			for j in range(M):
				cnd1 = i / N < 1/4 and j / M > 1/2
				cnd2 = i / N < 1/4 and j / M < 1/4
				cnd3 = i / N > 1/2 and j / M < 1/4
				cnd4 = i / N > 3/4 and 1/4 <= j / M < 1/2
				cnd5 = i / N > 1/2 and 1/2 <= j / M < 3/4

				# if outside of F
				if (cnd1 or cnd2 or cnd3 or cnd4 or cnd5):
					g[i,j] = rho[0]
				# if inside of F
				else:
					g[i,j] = rho[1]
	else:
		raise Exception(f'Distribution {dist} has not been implemented.')

	return g

# Double transform solution
def two_trans(g, which=1):

	def make_k(n, m, N, M):
		if which == 1:
			return np.pi**2 * (n**2 + m**2)
		if which == 2:
			return - 2 * M**2 * (np.cos(np.pi * n / N) + np.cos(np.pi * m / M) - 2)

	# get dimensions of g
	N, M = g.shape

	# double sine transform of g
	# save back in g
	g = fft.dstn(g, overwrite_x=True)

	# solve equation in transform space
	# save back in g
	g = - g / np.fromfunction(lambda n, m: make_k(n+1, m+1, N, M), (N, M))

	# double sine transform back
	# save in g again
	g = fft.idstn(g, overwrite_x=True)

	return g

# Double transform by hand
def two_trans_poor(g):

	def make_k(n, m):
		return np.pi**2 * (n**2 + m**2)

	# get dimensions of g
	N, M = g.shape

	# two 1d sine transforms of g
	# for efficiency reasons the transform is again saved in g
	for i in range(N):
		g[i,:] = fft.dst(g[i,:], overwrite_x=True)
	for j in range(M):
		g[:,j] = fft.dst(g[:,j], overwrite_x=True)

	# solve equation in transform space
	# for efficiency reasons save solution in g
	g = - g / np.fromfunction(lambda n, m: make_k(n+1, m+1), (N, M))

	# two 1d sine transforms back
	for j in range(M):
		g[:,j] = fft.idst(g[:,j], overwrite_x=True)
	for i in range(N):
		g[i,:] = fft.idst(g[i,:], overwrite_x=True)

	return g

# SOR method for square domain
def SOR(x, y, q, u, w, tol=10**(-5), max_itr=700, diffs=False, itrs=False):

	# confirm that sizes match
	assert len(x) == q.shape[0]
	assert len(x) == u.shape[0]
	assert len(y) == q.shape[1]
	assert len(y) == u.shape[1]

	# name sizes of array
	n, m = u.shape

	# difference in solutions
	def diff(u, v):
		n, m = u.shape
		return np.sum((u - v)**2) / (n * m)

	# initialize list of differences if necessary
	if diffs: difs = []

	# step size h and k and useful value d
	h = x[1] - x[0]
	k = y[1] - y[0]
	d = 1/2 * (h * k)**2 / (h**2 + k**2)

	# initialize difference, iteration number and start iteration loop	
	dif = float('inf')
	itr = 0
	while dif > tol and itr < max_itr:
		print(f'Iteration {itr}')

		# copy current solution
		up = np.array(u)
		
		# loop over arrays
		for i in range(n):
			for j in range(m):
				if i != 0 and i != n - 1 and j != 0 and j != m - 1:
					# update solution as specified by SOR
					u[i,j] = (w * (-q[i,j] + (u[i+1,j] + u[i-1,j])/h**2
						+ (u[i,j+1] + u[i,j-1])/k**2) * d + (1 - w) * u[i,j])
		
		# calculate difference in solution
		dif = diff(u, up)
		if diffs: difs.append(dif)

		# update iteration counter
		itr += 1

	if dif > tol:
		print('Iteration limit exceded.')
		print('Your solution may not be up to tolerance.')
		print('dif = ', dif)

	if diffs: return u, difs
	elif itrs: return u, itr
	else: return u

def find_optimal_SOR(x, y, q, u0, m_tol=10**(-8), w_a=None, w_pm=0.1,
	w_tol=10**(-3)):
	
	# module for search algorithm
	import search

	# default guess for optimal w if not specified
	if w_a == None:
		h = x[1] - x[0]
		k = y[1] - y[0]
		d = (h * k)**2 / (h**2 + k**2)

		rho = d * (np.cos(np.pi/len(x)) / h**2 + np.cos(np.pi/len(y)) / k**2)
		w_a = 2 / (1 + np.sqrt(1 - rho**2))

	# function to minimize 
	def itrs(w):
		_, itr = SOR(x, y, q, u0, w=w, tol=m_tol, itrs=True)
		return itr

	# minimize function with golden section search
	a, b = search.gss(itrs, w_a - w_pm, w_a + w_pm, tol=w_tol)

	return (b + a) / 2


# Single transform + discretization solution
def one_trans(g):

	# get dimensions of g
	N, M = g.shape

	# one 1d sine transform of g
	# for efficiency reasons save result back in g
	for j in range(M):
		g[:,j] = fft.dst(g[:,j], overwrite_x=True)

	# solve each row separately
	for i in range(N):

		# create banded form of matrix system
		d = - 4 + 2 * np.cos((i + 1) * np.pi / N)
		A_band = np.array([1, d, 1]).reshape((3,1)) * np.ones((3,M))

		# solve matrix system
		# for efficiency reasons save result in g again
		g[i,:] = la.solve_banded((1,1), A_band, g[i,:] / M**2, overwrite_ab=True,
			overwrite_b=True)

	# one 1d sine transform back
	for j in range(M):
		g[:,j] = fft.idst(g[:,j], overwrite_x=True)

	return g

def draw_profile(u, dist, rho=None, levels=100, savefig=False, showfig=True):
	N, M = u.shape

	fig, ax = plt.subplots(1)

	if rho is None: ttl = fr'Hang Profile, $N={N}$, $M={M}$'
	else: ttl = fr'Sag Profile, $\tilde{{\rho}}={rho[0]}$, $N={N}$, $M={M}$'

	ax.set_title(ttl)
	ax.set_xlabel(r'$x$ coordinate')
	ax.set_ylabel(r'$y$ coordinate')
	ax.set_aspect('equal')

	plt.plot(bounds[dist][0], bounds[dist][1], color='k')

	X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, M))
	plt.contour(X, Y, u, levels=levels//10, colors='k', linestyles='dashed', linewidths=1)
	cnt = plt.contourf(X, Y, u, levels=levels)
	
	clb = plt.colorbar(cnt)
	clb.ax.set_title(r'Sag $u$')

	if savefig:
		if rho is None: plt.savefig(f'images/profile_{dist}_{N}_{M}.png')
		else: plt.savefig(f'images/profile_{dist}_{rho[0]}_{N}_{M}.png')
	if showfig: plt.show()

"""
N = 100
M = 100
dist = 'hom'
rho = [10, 1]

g = make_g(N, M, dist, rho)
u = two_trans(g)

draw_profile(u, dist, rho)

print(np.min(u)) 
"""
"""
num_lst = np.geomspace(500, 70000, 100).astype(int)
#num_lst = range(12, 26)
times = []

for N in num_lst:

	print(f'Calculating for N=M={N}')

	t = timeit.timeit(stmt="fft.dst(g)", setup=f"import numpy as np; import scipy.fft as fft; g = np.random.rand({N})", number=10)
	times.append(t)

print('x:', list(num_lst))
print('y:', times)

xlabel = r'Number of 1D Points $\ln(N)$'
ylabel = r'Evaluation Time $\ln(t/[s])$'
title = fr'Evaluation Time for Different Numbers of Points'
#ylim = [-6, 3.9]
savepath = 'images/times_22p1.png'
#textbox = {'params': [0], 'symbols': ['slope'], 'coords': [0.6, 0.3]}
fig, ax = FitterPlotter('lin', [1.0, 1.0], np.log(num_lst), np.log(times),
	nstd=1, title=title, xlabel=xlabel, ylabel=ylabel, ret_fig=True,
	datalabel='dstn')
"""

#------------------------------------------------------------------------------
# Testing speed of implementation
# Very weird results???? it shouldn't be faster than NlogN.
dist = 'hom'

# different N
num_lst = np.geomspace(50, 500, 500-50).astype(int)
#num_lst = [2**N for N in range(6, 14)]

# Initizalize list
times_2st = []
times_2st_poor = []
times_1st = []
times_0st = []

for N in num_lst:

	print(f'Calculating for N=M={N}')

	g = np.random.rand(N,N)
	t_loc = []
	for _ in range(10):
		t = timeit.timeit(stmt="two_trans(g)", number=1, globals=globals())
		t_loc.append(t)
	times_2st.append(np.median(t_loc))

	g = np.random.rand(N,N)
	t_loc = []
	for _ in range(10):
		t = timeit.timeit(stmt="two_trans_poor(g)", number=1, globals=globals())
		t_loc.append(t)
	times_2st_poor.append(np.median(t_loc))

	g = np.random.rand(N,N)
	t_loc = []
	for _ in range(10):
		t = timeit.timeit(stmt="one_trans(g)", number=1, globals=globals())
		t_loc.append(t)
	times_1st.append(np.median(t_loc))



print('2ST good:', times_2st)
print('2ST poor:', times_2st_poor)
print('1ST:', times_1st)

xlabel = r'Number of 1D Points $\ln(N)$'
ylabel = r'Evaluation Time $\ln(t/[s])$'
title = fr'Evaluation Time for Different Numbers of Points'
#ylim = [-6, 3.9]
savepath = 'images/times_22p1_aalltt.png'
#textbox = {'params': [0], 'symbols': ['slope'], 'coords': [0.6, 0.3]}
fig, ax = FitterPlotter('lin', [1.0, 1.0], np.log(num_lst), np.log(times_2st),
	nstd=1, title=title, xlabel=xlabel, ylabel=ylabel, ret_fig=True,
	datalabel='2ST', showfig=False)

FitterPlotter('lin', [1.0, 1.0], np.log(num_lst), np.log(times_2st_poor), nstd=1,
	datalabel='2ST poor', figax=(fig, ax), showfig=False)

FitterPlotter('lin', [1.0, 1.0], np.log(num_lst), np.log(times_1st),
	nstd=1, datalabel='1ST', figax=(fig, ax), savepath=savepath)

#------------------------------------------------------------------------------
"""
#------------------------------------------------------------------------------
# Maximal sag as a function of density
dist = 'F'
N = 1000

rho_lst = np.geomspace(0.0001, 10000, 1000)
umin_lst = []
uminpos_lst = []

for rho in rho_lst:

	print(f'Calculating for rho={rho}.')

	g = make_g(N, N, dist, rho=[rho, 1])

	u = two_trans(g)

	umin_lst.append(np.min(u))
	uminpos_lst.append(np.array(np.unravel_index(np.argmin(u, axis=None), u.shape)))

uminpos_lst = np.array(uminpos_lst)

uder = np.diff(umin_lst)/np.diff(rho_lst)

fig, ax = plt.subplots(1)

ax.set_title(fr'Position of Maximal Sag, $N={N}$, $M={N}$')
ax.set_xlabel(r'$x$ coordinate')
ax.set_ylabel(r'$y$ coordinate')
ax.set_aspect('equal')
ax.set_ylim(0.37, 0.73)
ax.set_xlim(0.42, 0.61)

ax.plot(bounds[dist][0], bounds[dist][1], color='k')


sct = plt.scatter(uminpos_lst[:,1] / N, uminpos_lst[:,0] / N, c=np.log(rho_lst))
clb = plt.colorbar(sct)
clb.ax.set_title(r'Density $\ln(\tilde{\rho})$')

#plt.savefig(f'images/uminpos_{dist}_{N}_{N}.png')
plt.show()


fig, ax = plt.subplots(1)

ax.set_title(fr'Derivative of Maximal Sag over Densities, $N={N}$, $M={N}$')
ax.set_xlabel(r'Shaded Density $\tilde{\rho}$')
ax.set_ylabel(r'Maximal Sag Derivative $d u_{\mathrm{max}} / d\tilde{\rho}$')
ax.set_xscale('log')

plt.plot(rho_lst[:-1], uder, 'ko', markersize=2, label='Data')
plt.plot(rho_lst[:-1], uder)

plt.legend()
#plt.savefig(f'images/uminrho_{dist}_{N}_{N}.png')
plt.show()


fig, ax = plt.subplots(1)

ax.set_title(fr'Maximal Sag as a function of Density, $N={N}$, $M={N}$')
ax.set_xlabel(r'Shaded Density $\tilde{\rho}$')
ax.set_ylabel(r'Maximal Sag $|u_{\mathrm{max}}|$')
ax.set_xscale('log')
ax.set_yscale('log')

plt.plot(rho_lst, np.abs(umin_lst), 'ko', markersize=2, label='Data')
plt.plot(rho_lst, np.abs(umin_lst))

plt.legend()
plt.savefig(f'images/uminrho_notder_{dist}_{N}_{N}.png')
plt.show()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Plotting the profile of the sag of the F membrane
N = 1000
rho = [1, 1]

g = make_g(N, N, 'F', rho)
u = two_trans(g)

pos = np.unravel_index(np.argmin(u, axis=None), u.shape)

print(pos, u[pos[1], pos[0]])
draw_profile(u, 'F', rho=rho, savefig=True, showfig=False)
plt.plot(pos[1]/N, pos[0]/N, 'ro')
plt.show()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Plotting the profile of the sag of the homegeneous membrane
def make_odd(x):
	for i in range(len(x)):
		if x[i] % 2 == 0:
			x[i] = x[i] + 1

num_lst = np.geomspace(50, 5000, 100).astype(int)

umins1 = []
umins2 = []

for N in num_lst:
	print(f'Calculating for N=M={N}')

	g = make_g(N, N, 'hom')
	u = two_trans(g)

	umins1.append(np.min(u) + 0.0736713532814)

make_odd(num_lst)
for N in num_lst:
	print(f'Calculating for N=M={N}')

	g = make_g(N, N, 'hom')
	u = two_trans(g)

	umins2.append(np.min(u) + 0.0736713532814)


fig, ax = plt.subplots(1)
ax.set_xlabel(r'Number of 1D Points $N$')
ax.set_ylabel(r'Deviation of Maximal Sag $|u_{\mathrm{max}}^{\mathrm{num}} - u_{\mathrm{max}}|$')
ax.set_title(fr'Deviation of Maximal Sag from Analytical Value')
ax.set_yscale('log')
ax.grid()

plt.plot(num_lst, np.abs(umins1), label='mixed')
plt.plot(num_lst, np.abs(umins2), label='all odd')

plt.legend()
plt.savefig('images/conv.png')
plt.show()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Errors in the numerical solution
dist = 'hom'

num_lst = np.geomspace(50, 5000, 100).astype(int)

u1min = []
u2min = []
for N in num_lst:
	print(f'Calculating for N=M={N}')

	g = np.ones((N,N))
	u1 = two_trans(g, which=1)

	g = np.ones((N,N))
	u2 = two_trans(g, which=2)

	u1min.append(np.min(u1) + 0.0736713532814)
	u2min.append(np.min(u2) + 0.0736713532814)

fig, ax = plt.subplots(1)
ax.set_xlabel(r'Number of 1D Points $N$')
ax.set_ylabel(r'Deviation of Maximal Sag $|u_{\mathrm{max}}^{\mathrm{num}} - u_{\mathrm{max}}|$')
ax.set_title(fr'Deviation of Maximal Sag from Analytical Value')
ax.set_yscale('log')
ax.grid()

plt.plot(num_lst, np.abs(u1min), label='continuous')
plt.plot(num_lst, np.abs(u2min), label='discrete')

plt.legend()
plt.savefig('images/contvdis.png')
plt.show()
#------------------------------------------------------------------------------
"""





