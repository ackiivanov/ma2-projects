import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Convert to x and y
def convert(phi, L=1):

	# interval length
	ds = L / (N + 1)

	# construct matrices to store x and y
	x = np.empty(phi.shape)
	y = np.empty(phi.shape)
	
	# starting position
	x_cur = ds * np.cos(phi[0,:])
	y_cur = ds * np.sin(phi[0,:])

	# repeat for all internal points
	for i in range(phi.shape[0]):
		x[i] = x_cur
		y[i] = y_cur

		# if on the last point stop calculating
		if i == phi.shape[0]-1: continue

		x_cur += ds * (np.cos(phi[i,:]) + np.cos(phi[i+1,:])) / 2
		y_cur += ds * (np.sin(phi[i,:]) + np.sin(phi[i+1,:])) / 2

	return x, y


# Function that solves for the motion of the string
def solve_string(N, M, r, phi0, m=10, L=1):

	# check whether phi0 is of right length
	assert len(phi0) == N + 1

	# interval length
	ds = L / (N + 1)

	# banded matrix for F tridiagonal system
	def make_AFb(phin):
		AFb = np.empty((3, N+1))
		
		AFb[0,:] = np.ones(N+1)
		AFb[1,0] = -1
		AFb[1,1:-1] = -2 - (phin[2:] - phin[:-2])**2 / 4
		AFb[1,-1] = -1 - ds / m
		AFb[2,:] = np.ones(N+1)

		return AFb

	# create matrices for storing solution
	F = np.empty((N+1, M+1))
	phi = np.empty((N+1, M+1))

	# set initial condition for phi
	phi[:,0] = phi0

	# solve for initial condition for F
	# construct RHS of matrix equation
	b = np.zeros(N+1)
	b[0] = - ds * np.sin(phi0[0])

	# construct matrix
	AFb = make_AFb(phi0)
	
	# solve matrix equation
	F[:,0] = la.solve_banded((1,1), AFb, b)

	# time iterations
	for n in range(1, M+1):

		# updating phi of interior points
		if n == 1:
			phi[1:-1,n] = phi[1:-1,n-1] + (r**2 *
				((F[2:,n-1] - F[:-2,n-1]) * (phi[2:,n-1] - phi[:-2,n-1]) / 2 +
				F[1:-1,n-1] * (phi[2:,n-1] - 2 * phi[1:-1,n-1] + phi[:-2,n-1])))

		else:
			phi[1:-1,n] = 2 * phi[1:-1,n-1] - phi[1:-1,n-2] + (r**2 *
				((F[2:,n-1] - F[:-2,n-1]) * (phi[2:,n-1] - phi[:-2,n-1]) / 2 +
				F[1:-1,n-1] * (phi[2:,n-1] - 2 * phi[1:-1,n-1] + phi[:-2,n-1])))

		# updating phi of boundary points WIP for what to do with F
		if F[0,n-1] == 0: phi[0,n] = np.pi / 2
		else: phi[0,n] = phi[2,n] + 2 * ds * np.cos(phi[1,n]) / F[0,n-1]
		phi[N,n] = phi[N-1,n]

		# updating F of interior points + left boundary
		# construct RHS of matrix equation
		b = np.empty(N+1)
		b[0] = - ds * np.sin(phi[0,n])
		b[1:-1] = -(1/r**2) * (phi[1:-1,n] - phi[1:-1,n-1])**2
		b[-1] = 0
		
		# construct matrix
		AFb = make_AFb(phi[:,n])
		
		# solve matrix equation
		F[:,n] = la.solve_banded((1,1), AFb, b)

	return F, phi


def animate(x, y, savepath=None):

	fig, ax = plt.subplots(1)

	ax.set_title(r'Motion of String')
	ax.set_xlabel(r'$x$ coordinate')
	ax.set_ylabel(r'$y$ coordinate')
	ax.set_xlim(1.05 * np.min(x), 1.05 * np.max(x))
	ax.set_ylim(1.05 * np.min(-y), np.max(-y))
	ax.set_aspect('equal')

	line, = ax.plot(x[:,0], -y[:,0])
	point, = ax.plot(x[-1,0], -y[-1,0], 'ro')

	def update_anim(f, line=line):
		line.set_xdata(x[:,f])
		line.set_ydata(-y[:,f])

		point.set_xdata(x[-1,f])
		point.set_ydata(-y[-1,f])

		return line,

	anim = FuncAnimation(fig, update_anim, frames=range(x.shape[1]),
		interval=20, blit=True, save_count=50)

	if savepath is not None: anim.save('anims/' + savepath, dpi=150, fps=60, writer='ffmpeg')
	plt.show()

def length(x, y):
	return np.sum((x[1:,:] - x[:-1,:])**2 + (y[1:,:] - y[:-1,:])**2, axis=0)

def energy(phi):

	# interval lengths
	ds = L / (N + 1)
	dt = r * ds

	# get x and y from phi
	x, y = convert(phi)

	# calculate linear speed from x and y
	v2 = ((x[:-1,:] - x[1:,:])**2 + (y[:-1,:] - y[1:,:])**2) / dt**2

	# calculate angular speed from phi
	phidot2 = (phi[:-1,:] - phi[1:,:])**2 / dt**2

	# calculate energy
	E = v2 / 2 + ds**2 * phidot2 / 24 - y[:-1]



N = 100
M = 3000
r = 0.1

phi0 = (np.pi/2 - 0.2) * np.ones(N+1)
#phi0 = np.pi/2 - np.linspace(0, np.pi/5, N+1)

F, phi = solve_string(N, M, r, phi0)
x, y = convert(phi)


animate(x, y, savepath='2_orig.mp4')

plt.plot(length(x,y))
plt.show()
 
