import numpy as np
import scipy.linalg as la
import scipy.fft as fft

import matplotlib.pyplot as plt
from matplotlib import rc
import animatplot as amp


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

###############################################################################
# QUARANINED CODE THAT SOMEHOW STILL WORKS. DON'T TOUCH!!!!!!
def solve_nse_weird(T, Nt, N, Re, u0=1):
	#T = 0.01
	#Nt = 2000
	#N = 50
	#Re = 0.1

	dx = 1 / N
	dt = T / Nt

	print('dt/dx = ', np.around(dt/dx,6))
	
	u =    np.zeros((N+1,N+1,Nt+1))
	v =    np.zeros((N+1,N+1,Nt+1))
	zeta = np.zeros((N+1,N+1,Nt+1))
	psi =  np.zeros((N+1,N+1,Nt+1))

	for i in range(N+1):
		zeta[i,0,0] = - u0 / (dx)
		#zeta[i,1,0] = - u0 / (2*dx)
		#psi[i,1,0] = zeta[0,i,0]* dx*dx /2 + dx*u0

	for i in range(N+1):
		u[i,0,0] = u0


	def u_koef_2D(g_mreza, Nx, Ny, a, b):  #resi Poissonovo enacbo z DST
		h = a / (Nx-1)
	
		g_dst = fft.dstn(g_mreza)
	
		def koef(i,j,Nx=Nx,h=h,Ny=Ny):
			return 0.5 * (h**2) / (np.cos((i+1)*np.pi/Nx) + np.cos((j+1)*np.pi/Ny) - 2)

		u_dst = g_dst * np.fromfunction(koef, (Nx, Ny))
	
		u_mreza = fft.idstn(u_dst) 
	
		return u_mreza

	def iteracija_SOR(w, dx, N, u0, tol, ze):  
		h = dx
		u_t = np.zeros((N+1,N+1))
		u_z = np.zeros((N+1,N+1))

		stevec = 0  #stevec, koliko iteracij je ze poteklo
		razint = []
	
		while(1):
			#print('{}-ta iteracija  w={}'.format(stevec, w))
			stevec += 1

			for i in range(0, N+1):
				for j in range(0, N+1):       
					if i==0 or j==0 or i==N or j==N:
						u_t[i,j] = 0                 # prisilim nicle na robovih
					else:
						u_t[i,j] = (1-w)*u_z[i,j] + w*0.25*(h*h*ze[i,j] + u_z[i+1,j] + u_t[i-1,j] + u_z[i,j+1] + u_t[i,j-1])  #Gauss-Seidel
					'''
					elif j==1:
						u_t[i,1] = u_z[i,0] + dx*u0      
					elif j==N-1:
						u_t[i,N-1] = u_z[i,N] 
					elif i==1:
						u_t[1,j] = u_z[0,j] 
					elif i==N-1:
						u_t[N-1,j] = u_z[N,j] 
					'''
				

			raz = 0

			for i in range(0, N+1):
				for j in range(0, N+1):
					raz += (u_t[i,j] - u_z[i,j])**2   #izracunam vsoto kvadratov razlik med staro in novo resitvijo
					u_z[i,j] = u_t[i,j]  #nova resitev postane stara za naslednjo iteracijo

			razint.append(raz)
		
			if raz < tol:
				return [u_z, stevec, razint]


	for n in range(1,Nt+1):  #casovni razvoj resitve
		print('n = ',n)
		#plt.pcolormesh(zeta[:,:,n-1])
		#plt.colorbar()
		#plt.show()

		#w_opt = 2. / (1. + np.pi/N)
		#tol = 10**(-5.) 
		#[psi_n, stev, razint] = iteracija_SOR(w_opt, dx, N, u0, tol, zeta[n-1,:,:])

		psi_n = u_koef_2D(zeta[:,:,n-1], N+1, N+1, 1., 1.)

		for i in range(N+1):
			for j in range(N+1):
				psi[i,j,n] = psi_n[i,j]
	
		# iz psi ob casu n izracunamo hitrosti u in v ob casu n
		for i in range(1,N):
			for j in range(1,N):
				u[i,j,n] = (psi[i,j+1,n] - psi[i,j-1,n]) / (2*dx)
				v[i,j,n] = - (psi[i+1,j,n] - psi[i-1,j,n]) / (2*dx)
	
		for i in range(N+1): #hitrost na spodnjem robu
			u[i,0,n] = u0

		# izracunamo zeta ob casu n iz zete ob n-1 in hitrosti in psi ob n
		for i in range(1,N):  # po vseh notranjih tockah kvadrata
			for j in range(1,N):
				ox = u[i,j,n] * (zeta[i+1,j,n-1]  -  zeta[i-1,j,n-1]) / (2*dx)
				oy = v[i,j,n] * (zeta[i,j+1,n-1]  -  zeta[i,j-1,n-1]) / (2*dx)
				lap = (zeta[i+1,j,n-1] + zeta[i,j+1,n-1] + zeta[i-1,j,n-1] + zeta[i,j-1,n-1] - 4*zeta[i,j,n-1]) / dx**2
				zeta[i,j,n] = zeta[i,j,n-1] - dt*(ox + oy - lap/Re)

		for i in range(N+1): # po stranicah kvadrata
			zeta[i,N,n] = 2*psi[i,N-1,n] / dx**2
		
			zeta[0,i,n] = 2*psi[1,i,n]   / dx**2
			zeta[N,i,n] = 2*psi[N-1,i,n] / dx**2
			zeta[i,0,n] = 2*(psi[i,1,n] - u0*dx) / dx**2
		
	return u, v, zeta, psi
# QUARANINED CODE THAT SOMEHOW STILL WORKS. DON'T TOUCH!!!!!!
###############################################################################

# Animates a vector field
def animate_vec(u, v, Re, T, title='', savename=None):

	# get spatial and temporal number of steps
	N, _, M = u.shape

	# create grid
	x = np.linspace(0, 1, N)
	t = np.linspace(0, T, M)
	Y, X, T = np.meshgrid(x, x, t)

	# add title and labels
	plt.gca().set_aspect('equal')
	plt.gca().set_title(title)
	plt.gca().set_ylabel(r'Coordinate $y$')
	plt.gca().set_xlabel(r'Coordinate $x$')

	# animate vector field and add time axis
	block = amp.blocks.Quiver(X[:,:,0], Y[:,:,0], u, v, t_axis=2)
	anim = amp.Animation([block], amp.Timeline(t))
	anim.controls()

	# save and show
	if savename is not None: anim.save('anims/' + savename, dpi=150)
	plt.show()

# Animate a scalar field
def animate_scl(z, Re, T, title='', savename=None):

	# get spatial and temporal number of steps
	N, _, M = z.shape
	
	# create grid
	x = np.linspace(0, 1, N)
	t = np.linspace(0, T, M)
	Y, X, T = np.meshgrid(x, x, t)

	# add title and labels
	plt.gca().set_aspect('equal')
	plt.gca().set_title(title)
	plt.gca().set_ylabel(r'Coordinate $y$')
	plt.gca().set_xlabel(r'Coordinate $x$')

	# animate scalar field and add time axis
	block = amp.blocks.Pcolormesh(X[:,:,0], Y[:,:,0], z, t_axis=2, cmap='RdBu')
	plt.colorbar(block.quad)
	anim = amp.Animation([block], amp.Timeline(t))
	anim.controls()
	
	# save and show
	if savename is not None: anim.save('anims/' + savename, dpi=150)
	plt.show()

# Plot the convergence properties
def conv_plot(zeta, psi, Re, T, savename=None):

	# time size
	_, _, M = zeta.shape

	# calculate relative change
	zeta_rel = (np.sum(np.abs(zeta[:,:,1:] - zeta[:,:,:-1]), axis=(0,1))
		/ np.sum(np.abs(zeta[:,:,1:]), axis=(0,1)))
	psi_rel = (np.sum(np.abs(psi[:,:,1:] - psi[:,:,:-1]), axis=(0,1))
		/ np.sum(np.abs(psi[:,:,1:]), axis=(0,1)))

	# create figure and axes
	fig, ax = plt.subplots()

	# add title and labels
	ax.set_title(fr'Convergence Properties, $Re={Re}$')
	ax.set_ylabel(r'Relative Change $\sum_{ij}|Q^n_{ij} - Q^{n-1}_{ij}| / \sum_{ij} |Q^n_{ij}|$')
	ax.set_xlabel(r'Time $t$')
	ax.set_yscale('log')
	ax.grid()

	# plot the relative changes
	ax.plot(np.linspace(0, T, M-1), zeta_rel, label=r'Vorticity $\zeta$')
	ax.plot(np.linspace(0, T, M-1), psi_rel, label=r'Steam Function $\psi$')
	
	# save and show
	ax.legend()
	if savename is not None: fig.savefig(f'images/' + savename)
	plt.show()

# Plot the Flowline Profile
def stream_plot(u, v, Re, savename=None, ax=None):

	# decide whether to show plot
	if ax is None: showing = True
	else: showing = False

	# step number
	N, _ = u.shape

	# create grid
	x = np.linspace(0, 1, N)
	X, Y = np.meshgrid(x, x)

	# create figure and axes
	if ax is None: fig, ax = plt.subplots()

	# add title and labels
	ax.set_title(fr'Flowline Profile, $Re={Re}$')
	ax.set_ylabel(r'Coordinate $y$')
	ax.set_xlabel(r'Coordinate $x$')
	ax.set_xlim(-0.05, 1.05)
	ax.set_ylim(-0.05, 1.05)
	ax.set_aspect('equal')
	ax.grid()

	# plot the stream profile
	ax.plot([0,1,1,0,0], [0,0,1,1,0], 'k-')
	flw = plt.streamplot(X, Y, u.T, v.T)

	# save and show
	if savename is not None: fig.savefig(f'images/' + savename)
	if showing: plt.show()

# Solving the Navier-Stokes equation
def solve_nse(N, M, r, Re, ret_all=False, in_or_out='in'):

	# intervals
	dx = 1 / N
	dt = r * dx

	# indicate simulation parameters
	print(f'Solving equations using N={N}, M={M}, r={r}.')
	print(f'---> dx={dx}, dt={dt}.')
	print(f'Period of simulation is T={M * dt}.')

	# initialize arrays for solution
	#zeta = np.empty((N+1, N+1, M+1))
	#psi = np.empty((N+1, N+1, M+1))
	#u = np.empty((N+1, N+1, M+1))
	#v = np.empty((N+1, N+1, M+1))

	zeta = np.zeros((N+1, N+1, M+1))
	psi = np.zeros((N+1, N+1, M+1))
	u = np.zeros((N+1, N+1, M+1))
	v = np.zeros((N+1, N+1, M+1))

	# solve for current psi given zeta
	def solve_psi(n, tzeta):
		"""2DFFT for Dirichlet boundary conditions Poisson equation"""
	
		def make_k(n, m, N, which=2):
			if which == 1:
				return (-1) * np.pi**2 * (n**2 + m**2)
			if which == 2:
				return 2 * N**2 * (np.cos(np.pi * n / N) + np.cos(np.pi * m / N) - 2)

		# double sine transform
		g = fft.dstn(tzeta)

		# solve equation in transform space
		# save back in g
		g = g / np.fromfunction(lambda i, j: make_k(i+1, j+1, N+1), (N+1, N+1))

		# double sine transform back
		psi[:,:,n] = fft.idstn(g, overwrite_x=True)

	# solve for current velocity given psi
	def solve_uv(n, tpsi):

		# calculate u with bc u=1 at y=0
		u[:,0,n] = 1
		u[:,1:-1,n] = (tpsi[:,2:] - tpsi[:,:-2]) / (2 * dx)
		u[:,-1,n] = 1
		#u[0,:,n] = 0
		#u[-1,:,n] = 0

		# calculate v
		#v[0,:,n] = 0
		v[1:-1,:,n] = - (tpsi[2:,:] - tpsi[:-2,:]) / (2 * dx)
		#v[-1,:,n] = 0
		#v[:,0,n] = 0
		#v[:,-1,n] = 0

	# solve for current zeta given previos data
	def solve_zeta(n, tzeta, tu, tv, tpsi):

		# update interior points by formula, uv inside
		if in_or_out == 'in':
			term1 = tu[1:-1,1:-1] * (tzeta[2:,1:-1] - tzeta[:-2,1:-1]) / (2 * dx)
			term2 = tv[1:-1,1:-1] * (tzeta[1:-1,2:] - tzeta[1:-1,:-2]) / (2 * dx)
			term3 = (tzeta[2:,1:-1] + tzeta[:-2,1:-1] + tzeta[1:-1,2:] + tzeta[1:-1,:-2] - 4 * tzeta[1:-1,1:-1]) / (dx**2)
			zeta[1:-1,1:-1,n] = tzeta[1:-1,1:-1] - dt * (term1 + term2 - term3 / Re)
	
		# update interior points by formula, uv outside
		elif in_or_out == 'out':
			term1 = (tu[2:,1:-1] * tzeta[2:,1:-1] - tu[:-2,1:-1] * tzeta[:-2,1:-1]) / (2 * dx)
			term2 = (tv[1:-1,2:] * tzeta[1:-1,2:] - tv[1:-1,:-2] * tzeta[1:-1,:-2]) / (2 * dx)
			term3 = (tzeta[2:,1:-1] + tzeta[:-2,1:-1] + tzeta[1:-1,2:] + tzeta[1:-1,:-2] - 4 * tzeta[1:-1,1:-1]) / (dx**2)
			zeta[1:-1,1:-1,n] = tzeta[1:-1,1:-1] - dt * (term1 + term2 - term3 / Re)

		# update boundary points
		zeta[0,:,n] = 2 * tpsi[1,:] / dx**2
		zeta[-1,:,n] = 2 * tpsi[-2,:] / dx**2
		zeta[:,-1,n] = 2 * (tpsi[:,-2] - dx) / dx**2
		zeta[:,0,n] = 2 * (tpsi[:,1] - dx) / dx**2
		

	# initial conditions for velocity
	#u[:,1:,0] = 0
	#v[:,:,0] = 0
	u[:,0,0] = 1
	u[:,-1,0] = 1

	# initial conditions for psi
	#psi[:,:,0] = 0

	# initial conditions for zeta
	zeta[:,0,0] = - 1 / dx
	zeta[:,-1,0] = - 1 / dx
	#zeta[:,1:,0] = 0

	# interate time intervals
	for n in range(1,M+1):

		# Progress bar (69 can be whatever integer)
		print('#'*(69*n//M), end="\r", flush=True)

		# check if it has diverged
		for i, x in enumerate([u[:,:,n], v[:,:,n], zeta[:,:,n], psi[:,:,n]]):
			if np.isnan(np.min(x)):
				print(x)
				raise Exception(f'{i} has a nan at step {n}.')

		# solve for psi
		solve_psi(n, zeta[:,:,n-1])

		# solve for u and v
		solve_uv(n, psi[:,:,n])

		# solve for next zeta
		solve_zeta(n, zeta[:,:,n-1], u[:,:,n], v[:,:,n], psi[:,:,n])

	if ret_all: return u, v, zeta, psi
	else: return u, v

# Calculate the force on the lid
def lid_force(zeta, Re):

	# number of points
	N, _, _ = zeta.shape

	# integral of zeta
	intg = np.sum(zeta[:,0,:], axis=0) - (zeta[0,0,:] + zeta[-1,0,:]) / 2

	# force in the x direction
	return intg / (N * Re)

# Plot the force on the lid as a function of Re
def forceRe_plot(Re_lst, f, ylabel='', savename=None):

	# create figure and axes
	fig, ax = plt.subplots()

	# add title and labels
	ax.set_title(fr'Force $f_x$ on Lid as a function of $Re$')
	ax.set_ylabel(ylabel)
	ax.set_xlabel(r'Reynolds Number $Re$')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.grid()

	# plot the force
	ax.plot(Re_lst, f, 'ko')
	
	# save and show
	#ax.legend()
	if savename is not None: fig.savefig(f'images/' + savename)
	plt.show()

# Plot the force on the lid as a function of time
def forcet_plot(t, f, savename=None):

	# create figure and axes
	fig, ax = plt.subplots()

	# add title and labels
	ax.set_title(fr'Force $f_x$ on Lid as a function of Time')
	ax.set_ylabel(r'Tangential Force $f_x$')
	ax.set_xlabel(r'Time $t$')
	ax.grid()

	# plot the force
	ax.plot(t, f)

	# add inset axis
	axin = fig.add_axes([0.35, 0.19, 0.5, 0.45])
	axin.set_title(r'$|f_x - f_x^{\mathrm{final}}|$, log scale')
	axin.set_yscale('log')
	axin.grid()

	axin.plot(t[:-1], np.abs(f[:-1] - f[-1]))

	# save and show
	if savename is not None: fig.savefig(f'images/' + savename)
	plt.show()

# Plot a vector field at a given moment
def vec_plot(u, v, Re, savename=None, ax=None):

	# decide whether to show plot
	if ax is None: showing = True
	else: showing = False
	
	# time size
	N, _ = u.shape

	# create grid
	x = np.linspace(0, 1, N)
	X, Y = np.meshgrid(x, x)

	# create figure and axes
	if ax is None: fig, ax = plt.subplots()

	# add title and labels
	ax.set_title(fr'Velocity Vector Field $\mathbf{{v}}(\mathbf{{r}})$, $Re={Re}$')
	ax.set_ylabel(r'Coordinate $y$')
	ax.set_xlabel(r'Coordinate $x$')
	ax.set_aspect('equal')

	# plot the stream profile
	ax.quiver(X, Y, u.T, v.T)

	# save and show
	if savename is not None: fig.savefig(f'images/' + savename)
	if showing: plt.show()

# Plot a scalar field at a given moment
def scl_plot(z, title='', savename=None, ax=None, fig=None):

	# decide whether to show plot	
	if ax is None: showing = True
	else: showing = False

	# time size
	N, _ = z.shape

	# create grid
	x = np.linspace(0, 1, N)
	X, Y = np.meshgrid(x, x)

	# create figure and axes
	if ax is None: fig, ax = plt.subplots()

	# add title and labels
	ax.set_title(title)
	ax.set_ylabel(r'Coordinate $y$')
	ax.set_xlabel(r'Coordinate $x$')
	ax.set_aspect('equal')

	# plot the stream profile
	pcm = ax.pcolormesh(X, Y, z.T, cmap='RdBu', shading='auto')
	if showing: fig.colorbar(pcm)

	# save and show
	if savename is not None: fig.savefig(f'images/' + savename)
	if showing: plt.show()

	return pcm

# WIP
# Make a collage of pics
def scl_collage(z, T, savename=None):

	# shape
	_, _, M = z.shape

	# create figure and axes
	fig, axs = plt.subplots(2,3)

	order = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]

	for n in range(len(order)):

		pcm = scl_plot(z[:,:,(n*(M-2)//6 + 1)], ax=axs[order[n]], fig=fig,
			title=f'Time {(n*(M-2)//6 + 1)*T/M:3g}')

	fig.colorbar(pcm)
	if savename is not None: fig.savefig(f'images/' + savename)
	plt.show()



# be careful of number choice
N = 80
M = 7300
r = 2.5 * 10**(-1)
Re = 3000

# time of calculation
T = M * r / N

# calculate solution
#u, v, zeta, psi = solve_nse(N, M, r, Re, ret_all=True)
#u, v, zeta, psi = solve_nse_weird(T, M, N, Re)

# animate solution
#title = fr'Velocity vector field $\mathbf{{v}}(\mathbf{{r}},t)$, $Re={Re}$'
#animate_vec(u[:,:,::55], v[:,:,::55], Re, T, title=title, savename=f'Re{Re}_N{N}_M{M}_r{r}_uv.mp4')


r"""
# animate solution
title = fr'Velocity vector field $\mathbf{{v}}(\mathbf{{r}},t)$, $Re={Re}$'
animate_vec(u[:,:,::10], v[:,:,::10], Re, T, title=title, savename=f'Re{Re}_N{N}_M{M}_r{r}_uv.mp4')
animate_vec(u, v, Re, T, title=title, savename=f'Re{Re}_N{N}_M{M}_r{r}_uv_test.mp4')

title = fr'Vorticity field $\zeta (\mathbf{{r}},t)$, $Re={Re}$'
animate_scl(zeta[:,:,::10], Re, T, title=title, savename=f'Re{Re}_N{N}_M{M}_r{r}_zeta.mp4')
title = fr'Stream function field $\psi (\mathbf{{r}},t)$, $Re={Re}$'
animate_scl(psi[:,:,::10], Re, T, title=title, savename=f'Re{Re}_N{N}_M{M}_r{r}_psi.mp4')
title = fr'Velocity magnitude field $\|\mathbf{{v}}\| (\mathbf{{r}},t)$, $Re={Re}$'
animate_scl(np.sqrt(u**2+v**2)[:,:,::10], Re, T, title=title,
	savename=f'Re{Re}_N{N}_M{M}_r{r}_vmag.mp4')
"""

"""
conv_plot(zeta, psi, Re, T, savename=f'Re{Re}_N{N}_M{M}_r{r}_conv.png')
conv_plot(zeta[:,:,:M//16], psi[:,:,:M//16], Re, T/16, savename=f'Re{Re}_N{N}_M{M}_r{r}_conv_osc.png')

forcet_plot(np.linspace(0, T, M+1), lid_force(zeta, Re),
	savename=f'Re{Re}_N{N}_M{M}_r{r}_forcet.png')

vec_plot(u[:,:,-1], v[:,:,-1], Re, savename=f'Re{Re}_N{N}_M{M}_r{r}_uv.png')
stream_plot(u[:,:,-1], v[:,:,-1], Re, savename=f'Re{Re}_N{N}_M{M}_r{r}_stream.png')

title = fr'Vorticity field $\zeta (\mathbf{{r}},t)$, $Re={Re}$'
scl_plot(zeta[:,:,-1], title=title, savename=f'Re{Re}_N{N}_M{M}_r{r}_zeta.png')
scl_plot(zeta[1:-1,1:-1,-1], title=title, savename=f'Re{Re}_N{N}_M{M}_r{r}_zeta_test.png')
title = fr'Stream function field $\psi (\mathbf{{r}},t)$, $Re={Re}$'
scl_plot(psi[:,:,-1], title=title, savename=f'Re{Re}_N{N}_M{M}_r{r}_psi.png')

title = fr'Vorticity Assymetry $\zeta (x,y) - \zeta (-x,y)$, $Re={Re}$'
scl_plot(zeta[:,:,-1] - zeta[::-1,:,-1], title=title, savename=f'Re{Re}_N{N}_M{M}_r{r}_zeta_asym.png')
title = fr'Stream function Asymmetry $\psi (x,y) - \psi (-x,y)$, $Re={Re}$'
scl_plot(psi[:,:,-1] - psi[::-1,:,-1], title=title, savename=f'Re{Re}_N{N}_M{M}_r{r}_psi_asym.png')

title = fr'Velocity magnitude field $\|\mathbf{{v}}\| (\mathbf{{r}})$, $Re={Re}$'
scl_plot(np.sqrt(u**2+v**2)[:,:,-1], title=title, savename=f'Re{Re}_N{N}_M{M}_r{r}_vmag.png')
"""



def sym_plot(logRe_L, logRe_H, Re_N, N, M, r, savename=None):

	# Symmetry as a function of Re
	Re_lst = np.logspace(logRe_L, logRe_H, Re_N)

	zeta_dif = []
	psi_dif = []

	for Re in Re_lst:
		_, _, zeta, psi = solve_nse(N, M, r, Re, ret_all=True)

		zeta_dif.append(np.max(np.abs(zeta[:,:,-1] - zeta[::-1,:,-1])))
		psi_dif.append(np.max(np.abs(psi[:,:,-1] - psi[::-1,:,-1])))

	# create figure and axes
	fig, ax = plt.subplots()

	# add title and labels
	ax.set_title(fr'Symmetry Property')
	ax.set_ylabel(r'Maximum Change $\mathrm{max}|Q(x,y) - Q(-x, y)|$')
	ax.set_xlabel(r'Reynolds Number $Re$')
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.grid()

	# plot the maximal changes
	ax.plot(Re_lst, zeta_dif, label=r'Vorticity $\zeta$')
	ax.plot(Re_lst, psi_dif, label=r'Steam Function $\psi$')
	
	# save and show
	ax.legend()
	if savename is not None: fig.savefig(f'images/' + savename)
	plt.show()	
	
#sym_plot(-1, 3, 20, N, M, r, savename=f'N{N}_M{M}_r{r}_sym.png')


def stime_plot(Re_lst, N, M_lst, r_lst, savename=None):

	def converged(zeta, eps=10**(-4)):
		# calculate relative change
		zeta_rel = (np.sum(np.abs(zeta[:,:,1:] - zeta[:,:,:-1]), axis=(0,1))
			/ np.sum(np.abs(zeta[:,:,1:]), axis=(0,1)))
		
		return np.argmax(zeta_rel < eps)
	
	T_lst = []

	for i in range(len(Re_lst)):
		_, _, zeta, _ = solve_nse(N, M_lst[i], r_lst[i], Re_lst[i], ret_all=True)

		T_lst.append(r_lst[i] * converged(zeta) / N)

	# create figure and axes
	fig, ax = plt.subplots()

	# add title and labels
	ax.set_title(fr'Time for Convergence')
	ax.set_ylabel(r'Time $T$')
	ax.set_xlabel(r'Reynolds Number $Re$')
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.grid()

	# plot the maximal changes
	ax.plot(Re_lst, T_lst, 'ko')
	
	# save and show
	if savename is not None: fig.savefig(f'images/' + savename)
	plt.show()	

N = 80
Re_lst = [0.1, 1, 10, 100, 1000, 3000]
r_lst = [10**(-4), 10**(-3), 10**(-2), 0.25, 0.25, 0.25]
M_lst = [1000, 2000, 3000, 7000, 10000, 11000]
#stime_plot(Re_lst, N, M_lst, r_lst, savename='stimes.png')


f = []
fRe = []
for i in range(len(Re_lst)):
	M = M_lst[i]
	r = r_lst[i]
	Re = Re_lst[i]

	T = M * r / N

	u, v, zeta, psi = solve_nse(N, M, r, Re, ret_all=True)
	
	title = fr'Velocity vector field $\mathbf{{v}}(\mathbf{{r}},t)$, $Re={Re}$'
	animate_vec(u[:,:,::55], v[:,:,::55], Re, T, title=title, savename=f'modRe{Re}_N{N}_M{M}_r{r}_uv.mp4')

	conv_plot(zeta, psi, Re, T, savename=f'modRe{Re}_N{N}_M{M}_r{r}_conv.png')
	conv_plot(zeta[:,:,:M//16], psi[:,:,:M//16], Re, T/16, savename=f'modRe{Re}_N{N}_M{M}_r{r}_conv_osc.png')

	forcet_plot(np.linspace(0, T, M+1), lid_force(zeta, Re),
		savename=f'modRe{Re}_N{N}_M{M}_r{r}_forcet.png')

	vec_plot(u[:,:,-1], v[:,:,-1], Re, savename=f'modRe{Re}_N{N}_M{M}_r{r}_uv.png')
	stream_plot(u[:,:,-1], v[:,:,-1], Re, savename=f'modRe{Re}_N{N}_M{M}_r{r}_stream.png')

	title = fr'Vorticity field $\zeta (\mathbf{{r}},t)$, $Re={Re}$'
	scl_plot(zeta[:,:,-1], title=title, savename=f'modRe{Re}_N{N}_M{M}_r{r}_zeta.png')
	scl_plot(zeta[1:-1,1:-1,-1], title=title, savename=f'modRe{Re}_N{N}_M{M}_r{r}_zeta_test.png')
	title = fr'Stream function field $\psi (\mathbf{{r}},t)$, $Re={Re}$'
	scl_plot(psi[:,:,-1], title=title, savename=f'modRe{Re}_N{N}_M{M}_r{r}_psi.png')

	title = fr'Velocity magnitude field $\|\mathbf{{v}}\| (\mathbf{{r}})$, $Re={Re}$'
	scl_plot(np.sqrt(u**2+v**2)[:,:,-1], title=title, savename=f'modRe{Re}_N{N}_M{M}_r{r}_vmag.png')
	
	#force = lid_force(zeta, Re_lst[i])

	#f.append(force[-1])
	#fRe.append(Re_lst[i] * force[-1])

#ylabel = r'Tangential Force $|f_x|$'
#forceRe_plot(Re_lst, np.abs(f), ylabel=ylabel, savename=f'forceRe1.png')
#ylabel = r'Scaled Tangential Force $|Re \, f_x|$'
#forceRe_plot(Re_lst, np.abs(fRe), ylabel=ylabel, savename=f'ReforceRe1.png')
