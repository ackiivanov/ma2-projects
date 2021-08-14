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
def solve_string(N, M, r, phi0, L=1, bc2=1):

	# check whether phi0 is of right length
	assert len(phi0) == N + 1

	# interval length
	ds = L / (N + 1)

	# banded matrix for F tridiagonal system
	def make_AFb(phin):
		AFb = np.empty((3, N))
		
		AFb[0,:] = np.ones(N)
		AFb[1,0] = -1
		AFb[1,1:] = -2 - (phin[2:] - phin[:-2])**2 / 4
		AFb[2,:] = np.ones(N)

		return AFb

	# create matrices for storing solution
	F = np.empty((N+1, M+1))
	phi = np.empty((N+1, M+1))

	# set initial condition for phi
	phi[:,0] = phi0

	# solve for initial condition for F
	# construct RHS of matrix equation
	b = np.zeros(N)
	b[0] = - ds * np.sin(phi0[0])

	# construct matrix
	AFb = make_AFb(phi0)
	
	# solve matrix equation
	F[:-1,0] = la.solve_banded((1,1), AFb, b)

	# add right boundary
	F[-1,0] = 0

	# time iterations
	for n in range(1, M+1):

		# updating phi of interior points
		if n == 1:
			phi[1:-1,n] = phi[1:-1,n-1] + (r**2 *
				((F[2:,n-1]- F[:-2,n-1]) * (phi[2:,n-1] - phi[:-2,n-1]) / 2 +
				F[1:-1,n-1] * (phi[2:,n-1] - 2 * phi[1:-1,n-1] + phi[:-2,n-1])))

		else:
			phi[1:-1,n] = 2 * phi[1:-1,n-1] - phi[1:-1,n-2] + (r**2 *
				((F[2:,n-1]- F[:-2,n-1]) * (phi[2:,n-1] - phi[:-2,n-1]) / 2 +
				F[1:-1,n-1] * (phi[2:,n-1] - 2 * phi[1:-1,n-1] + phi[:-2,n-1])))

		# updating phi of boundary points WIP for what to do with F
		if F[0,n-1] == 0: phi[0,n] = np.pi / 2
		else: phi[0,n] = phi[2,n] + 2 * ds * np.cos(phi[1,n]) / F[0,n-1]
		
		if bc2 == 1: phi[N,n] = 2 * phi[N-1,n] - phi[N-2,n]
		else: phi[N,n] = phi[N-1,n]

		# updating F of interior points + left boundary
		# construct RHS of matrix equation
		b = np.empty(N)
		b[0] = - ds * np.sin(phi[0,n])
		b[1:] = -(1/r**2) * (phi[1:-1,n] - phi[1:-1,n-1])**2
		
		# construct matrix
		AFb = make_AFb(phi[:,n])
		
		# solve matrix equation
		F[:-1,n] = la.solve_banded((1,1), AFb, b)

		# updating F at right boundary point
		F[N,n] = 0

	return F, phi


def animate(x, y, r, z=None, w=None, L=1, eta=0.25,
	xlabel=r'$x$ coordinate', ylabel=r'$y$ coordinate',
	equal=True, savepath=None):

	if z is not None: assert w is not None
	if w is not None: assert z is not None

	N, M = x.shape

	fig, ax = plt.subplots(1)

	ax.set_title(r'Motion of String')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_xlim(np.min(x),np.max(x))
	ax.set_ylim(np.min(-y), np.max(-y))
	if equal == True: ax.set_aspect('equal')

	if z is None: line, = ax.plot(x[:,0], -y[:,0])
	else:
		line, = ax.plot(x[:,0], -y[:,0], label='choice 1')
		line2, = ax.plot(z[:,0], -w[:,0], label='choice 2')

	def update_anim(f):
		line.set_xdata(x[:,f])
		line.set_ydata(-y[:,f])

		if z is not None:
			line2.set_xdata(z[:,f])
			line2.set_ydata(-w[:,f])

		return line,

	anim = FuncAnimation(fig, update_anim, frames=range(x.shape[1]),
		interval=20, blit=True, save_count=50)

	if z is not None: plt.legend(loc='upper right')
	if savepath is not None: anim.save('anims/' + savepath, dpi=150,
		fps=int(eta * (N + 1) / (L * r)), writer='ffmpeg')
	plt.show()

	return fig, ax

def length(x, y, L=1):
	# interval length
	# this is to compensate for the first and last half elements
	ds = L / x.shape[0]

	return np.sum(((x[1:,:] - x[:-1,:])**2 +
		(y[1:,:] - y[:-1,:])**2)**(1/2), axis=0) + ds

def energy(phi, x=None, y=None, L=1, norm=False):

	if x is None:
		x, y = convert(phi)

	# dimensions
	N = phi.shape[0]

	# interval lengths
	ds = L / N
	dt = r * ds

	# get x and y from phi
	x, y = convert(phi)

	# calculate linear speed from x and y
	v2 = ((x[:,1:] - x[:,:-1])**2 + (y[:,1:] - y[:,:-1])**2) / dt**2

	# calculate angular speed from phi
	phidot2 = (phi[:,1:] - phi[:,:-1])**2 / dt**2

	# calculate energy
	E = np.sum(v2 / 2 + ds**2 * phidot2 / 24 - y[:,:-1], axis=0)

	# normalize energy if necessary
	if norm: E = E / E[1] - 1

	print(E[0])

	return E[1:]

def draw_conf(A, kind='F', L=1, levels=7):

	# dimensions
	N, M = A.shape

	# array of lengths
	s = np.linspace(0, L, N)

	# plotting
	fig, ax = plt.subplots(1)

	if kind == 'F': ax.set_title(r'$F$ as a function of $s$ for a few times')
	if kind == 'phi': ax.set_title(r'$\phi$ as a function of $s$ for a few times')
	ax.set_xlabel(r'Length parameter $s$')
	if kind == 'F': ax.set_ylabel(r'Force $F(s)$')
	if kind == 'phi': ax.set_ylabel(r'Angle $\phi(s)$')
	ax.grid()
	
	# draw a few different 
	for l in range(levels):
		ind = int(l * M / levels)
		plt.plot(s, A[:,ind], label=fr'$n={ind}$')

	plt.legend()
	plt.show()


"""
N = 1000
M = 43000
r = 0.1

#phi0 = 0.1 * np.ones(N+1)
#phi0 = (np.pi/2 - 0.1) * np.ones(N+1)
phi0 = np.pi/2 - np.linspace(0, 1.2 * np.pi, N+1)
#phi0 = np.pi/2 - np.linspace(0, 0.6, N+1)

F, phi = solve_string(N, M, r, phi0)
x, y = convert(phi)

#savepath = 'bic2.mp4'
#animate(x[:,::10], y[:,::10], r, eta=0.005, savepath=savepath)

plt.plot(energy(phi, norm=True))
#plt.plot(range(M), [np.sum(-y[:,0])] * M)
plt.show()

plt.plot(np.std(phi, axis=0))
plt.show()

plt.plot(phi[0,:])
plt.show()


s = np.empty((N+1, M+1))
for n in range(M+1):
	s[:,n] = np.linspace(0, 1, N + 1)

#animate(s[:,::20], -F[:,::20], r, eta=0.01, savepath='bic2_F.mp4')
"""

#animate(x, y)
"""
plt.plot(length(x,y))
plt.show()


"""
"""
for n in range(M):
	plt.title(f'Angle {n}/{M}')
	plt.plot(phi[:,n])
	plt.show()

	plt.title(f'Force {n}/{M}')
	plt.plot(F[:,n])
	plt.show()
"""




"""
#------------------------------------------------------------------------------
# Testing the two boundary conditions at the free end

# parameters
N = 100
M = 15000
r = 0.1

# intial angles
phi0 = np.pi/2 - np.linspace(0, 0.6, N+1)

# solve with first choice of boundary condition
F1, phi1 = solve_string(N, M, r, phi0, bc2=1)
x1, y1 = convert(phi1)

# solve with second choice of boundary condition
F2, phi2 = solve_string(N, M, r, phi0, bc2=2)
x2, y2 = convert(phi2)

#savepath = '2bcs.mp4'
#animate(x1, y1, r, x2, y2, savepath=savepath)

draw_conf(np.abs(phi1 - phi2))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# For what angles does crossing happen?

# parameters
N = 200
M = 12600
r = 0.1

constphi_lst = np.linspace(np.pi/2, 0.18, 40)

ergs1 = []
ergs2 = []
for constphi in constphi_lst:

	print(f'Solving for constphi={constphi}.')

	phi0 = constphi * np.ones(N+1)

	F, phi = solve_string(N, M, r, phi0)

	E = energy(phi, norm=True)

	ergs1.append(np.max(E))
	ergs2.append(np.max(E[:M//2]))


fig, ax = plt.subplots(1)

ax.set_title(fr'Energy Deviation for Different $\phi_0$, $N={N}$, $M={M}$, $r={r}$')
ax.set_xlabel(r'Initial Constant Angle $\phi_0$')
ax.set_ylabel(r'Energy Deviation $\mathrm{max}_{t} (E/E_0 - 1) $')
ax.grid()

ax.plot(constphi_lst, ergs1, label=r'$\sim\!2$ swings')
ax.plot(constphi_lst, ergs2, label=r'$\sim\!1$ swing')

plt.legend()
plt.savefig(f'images/ergdev_phi0_{M}.png')
plt.show()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Profile of oscillations for a few different points

# parameters
N = 200
M = 300000
r = 0.01
levels = 7
initphi = 0.4

# intial angles
phi0 = initphi * np.ones(N+1)

# solve with first choice of boundary condition
F, phi = solve_string(N, M, r, phi0, bc2=1)
x, y = convert(phi)

# array of times
t = np.linspace(0, r * (M + 1) / (N + 1), M + 1)

# plotting phi
fig, ax = plt.subplots(1)

ax.set_title(fr'$\phi$ over time, $\phi_0={initphi:.3f}$, $N={N}$, $M={M}$, $r={r}$')
ax.set_xlabel(r'Time $t$')
ax.set_ylabel(r'Angle $\phi(t)$')
ax.grid()
	
# draw a few different 
for l in range(levels):
	ind = int(l * (N + 1) / levels)
	plt.plot(t, phi[ind,:], label=fr'$i={ind}$')

plt.legend(loc='lower right')
plt.savefig(f'images/phi_t_{initphi}.png')
plt.show()

# plotting F
fig, ax = plt.subplots(1)

ax.set_title(fr'$F$ over time, $\phi_0={initphi:.3f}$, $N={N}$, $M={M}$, $r={r}$')
ax.set_xlabel(r'Time $t$')
ax.set_ylabel(r'Force $F(t) - \langle F \rangle_t$')
ax.grid()
	
# draw a few different 
for l in range(levels):
	ind = int(l * (N + 1) / levels)
	plt.plot(t, F[ind,:] - np.average(F[ind,:]), label=fr'$i={ind}$')

plt.legend(loc='lower right')
plt.savefig(f'images/F_t_{initphi}.png')
plt.show()

# plotting E
fig, ax = plt.subplots(1)

ax.set_title(fr'$E$ over time, $\phi_0={initphi:.3f}$, $N={N}$, $M={M}$, $r={r}$')
ax.set_xlabel(r'Time $t$')
ax.set_ylabel(r'Energy $E$')
ax.grid()

ax.plot(t[1:-1], energy(phi, x, y))

plt.savefig(f'images/E_t_{initphi}.png')
plt.show()
#------------------------------------------------------------------------------
"""
#------------------------------------------------------------------------------
# Profile of string for a few different times

# parameters
N = 200
M = 126000
r = 0.01
levels = 7
initphi = 000000

# intial angles
#phi0 = initphi * np.ones(N+1)
phi0 = np.linspace(np.pi/2, -0.3, N+1)

# solve with first choice of boundary condition
F, phi = solve_string(N, M, r, phi0, bc2=1)

# array of lengths
s = np.linspace(0, 1, N+1)
ss = np.empty((N+1, M+1))
for n in range(M+1):
	ss[:,n] = s

# plotting phi
fig, ax = plt.subplots(1)

ax.set_title(fr'$\phi$ over string, $\phi_0={initphi:.3f}$, $N={N}$, $M={M}$, $r={r}$')
ax.set_xlabel(r'Length parameter $s$')
ax.set_ylabel(r'Angle $\phi(s)$')
ax.grid()
	
# draw a few different 
for l in range(levels):
	ind = int(l * (M + 1) / levels)
	plt.plot(s, phi[:,ind], label=fr'$n={ind}$')

plt.legend(loc='upper left')
plt.savefig(f'images/phi_s_{initphi}.png')
plt.show()

# animate
#xlabel = r'Length parameter $s$'
#ylabel = r'Angle $\phi$'
#savepath = 'phi.mp4'
#animate(ss[:,::20], -phi[:,::20], r, eta=0.005, xlabel=xlabel, ylabel=ylabel, savepath=savepath, equal=False)

# plotting F
fig, ax = plt.subplots(1)

ax.set_title(fr'$F$ over string, $\phi_0={initphi:.3f}$, $N={N}$, $M={M}$, $r={r}$')
ax.set_xlabel(r'Length parameter $s$')
ax.set_ylabel(r'Force $F(s)$')
ax.grid()
	
# draw a few different 
for l in range(levels):
	ind = int(l * (N + 1) / levels)
	plt.plot(s, F[:,ind], label=fr'$n={ind}$')

plt.legend(loc='upper right')
plt.savefig(f'images/F_s_{initphi}.png')
plt.show()

# animate
#xlabel = r'Length parameter $s$'
#ylabel = r'Force $F$'
#savepath = 'F.mp4'
#animate(ss[:,::20], -F[:,::20], r, eta=0.005, xlabel=xlabel, ylabel=ylabel, savepath=savepath, equal=False)
#------------------------------------------------------------------------------
"""
#------------------------------------------------------------------------------
# Forces as a function of initial angle

# parameters
N = 200
M = 100000
r = 0.01
print('T =', r * (M + 1) / (N + 1))
initphi_lst = np.linspace(np.pi / 2, 0.4, 40)

F0avg_lst = []
F0min_lst = []
F0max_lst = []
for initphi in initphi_lst:

	print(f'Calculating with initphi={initphi}.')

	# intial angles
	phi0 = initphi * np.ones(N+1)

	# solve with first choice of boundary condition
	F, phi = solve_string(N, M, r, phi0, bc2=1)

	#F0avg_lst.append(np.average(F[0,:]))
	#F0min_lst.append(np.min(F[0,:]))
	#F0max_lst.append(np.max(F[0,:]))

	F0avg_lst.append((F[0,-1] + F[0,0]) / 2)
	F0min_lst.append(F[0,0])
	F0max_lst.append(F[0,-1])


# plotting F0
fig, ax = plt.subplots(1)

ax.set_title(fr'$F_0$ over Initial Angles, $N={N}$, $M={M}$, $r={r}$')
ax.set_xlabel(r'Initial Angle $\phi_0$')
ax.set_ylabel(r'Force $F(s=0)$')
ax.grid()

line, = ax.plot(initphi_lst, F0avg_lst, label='Average')
ax.fill_between(initphi_lst, F0max_lst, F0min_lst, color=line.get_color(), alpha=0.25, label='Range')

plt.legend(loc='upper right')
plt.savefig('images/F0_phi0_M.png')
plt.show()
#------------------------------------------------------------------------------
"""









