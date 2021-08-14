import numpy as np
import scipy.linalg as la
import scipy.fft as fft
import scipy.stats as st

import matplotlib.pyplot as plt
from matplotlib import rc
from FitterPlotter import FitterPlotter

from timeit import default_timer as timer
 

# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# Single transform + discretization solution
def solve_profile(N, M, r_bc=None, src=None):

	if r_bc is None: r_bc = np.ones(N)

	if src is None: src = np.zeros((N,M))

	def make_A_band(m, M):

		# initialize banded form of matrix
		A = np.zeros((3,M))

		for i in range(3):
			for j in range(M):

				# implement Neumann boundary conditions
				if i == 0 and j == 1:
					A[i,j] = 1
				elif i == 1 and j == 0:
					A[i,j] = -1 - (m * np.pi / (2 * M))**2

				# implement the discretized Laplacian
				elif i == 0:
					A[i,j] = 1 + 1 / (2 * (j - 1))
				elif i == 1:
					A[i,j] = -2 - (m * np.pi / (2 * M))**2
				elif i == 2:
					A[i,j] = 1 - 1 / (2 * (j + 1))

		return A

	# initialize solution matrix
	u = np.zeros((N,M))

	# one 1d sine transform of Dirichlet boundary condition and source
	u_bc2 = fft.dst((-1) * r_bc)
	u_src = np.empty((M,N))
	for i in range(M):
		u_src[i] = fft.dst((-1) * src[:,i])
	u_src = u_src.T

	# solve each row separately
	for i in range(N):

		# create banded form of matrix system
		A_band = make_A_band(i + 1, M)

		# implement Dirichlet condition with vector
		b = u_src[i]
		b[-1] = u_bc2[i]

		# solve matrix system
		u[i,:] = la.solve_banded((1,1), A_band, b / M**2, overwrite_ab=True,
			overwrite_b=True)

	# one 1d sine transform back
	# save solution back into u
	for j in range(M):
		u[:,j] = fft.idst(u[:,j], overwrite_x=True) * M**2 # weird normalizaton???

	return u

def draw_profile(u, levels=100, savename=None, showfig=True):
	
	N, M = u.shape

	fig, ax = plt.subplots(1)
	ax.set_title(fr'Temperature Profile, $N={N}$, $M={M}$')
	ax.set_xlabel(r'$r$ coordinate')
	ax.set_ylabel(r'$z$ coordinate')
	ax.set_aspect('equal')

	X, Y = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 2, N))
	plt.contour(X, Y, u, levels=levels//10, colors='k', linestyles='dashed', linewidths=1)
	cnt = plt.contourf(X, Y, u, cmap='jet', levels=levels)
	
	clb = plt.colorbar(cnt)
	clb.ax.set_title(r'Temp. $T$')

	if savename is not None: fig.savefig('images/' + savename)
	if showfig: plt.show()

N = 5000
M = 5000


#r_bc = np.sin(np.linspace(0, 2*np.pi, N))

#r_bc = np.zeros(N)
#for i in range(N//2 - 100, N//2 + 100): r_bc[i] = 1 

s = np.linspace(-1, 1, N)
def normal(x, mu=0, sigma=1):
	return np.exp((-1) * (x - mu)**2 / (2 * sigma**2))
#r_bc = normal(s, mu=0, sigma=1/5) + normal(s, mu=1, sigma=1/5)
#r_bc = normal(s, mu=0, sigma=1/5)

#X, Y = np.mgrid[-2:2:1j*N, -2:2:1j*M]
#src = normal(np.sqrt(X**2 + Y**2), sigma=0.5)
src = np.zeros((N,M))
for n in range(N//3 - 100, N//3 + 100):
	for m in range(M//3 - 100, M//3 + 100):
		src[n,m] = 1

u = solve_profile(N, M, src=src)

draw_profile(u, savename=f'tempprofile_{N}_{M}_delta_src.png')

print(np.max(u))











