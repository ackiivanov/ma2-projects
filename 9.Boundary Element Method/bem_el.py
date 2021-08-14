import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
from matplotlib import rc
from FitterPlotter import FitterPlotter

from timeit import default_timer as timer


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# Green's function in CM frame
def G0(x, y, l):
	xm = x - l / 2
	xp = x + l / 2

	if y == 0 and xp == 0 and xm == 0:
		return 1 / (2 * np.pi) * (-l)
	if y == 0 and xm == 0:
		return 1 / (2 * np.pi) * (-l + xp * np.log(np.abs(xp)))
	if y == 0 and xp == 0:
		return 1 / (2 * np.pi) * (-l - xm * np.log(np.abs(xm)))
	if y == 0:
		return 1 / (2 * np.pi) * (-l + xp * np.log(np.abs(xp)) -
			xm * np.log(np.abs(xm)))
	if xm == 0:
		return 1 / (2 * np.pi) * (-l + y * np.arctan(xp / y) +
			xp / 2 * np.log(xp**2 + y**2))
	if xp == 0:
		return 1 / (2 * np.pi) * (-l - y * np.arctan(xm / y) -
			xm / 2 * np.log(xm**2 + y**2))

	return 1 / (2 * np.pi) * (-l + y * np.arctan(xp / y) - y * np.arctan(xm / y)
		+ xp / 2 * np.log(xp**2 + y**2) - xm / 2 * np.log(xm**2 + y**2))
G0 = np.vectorize(G0)

# parametrize boundary
def make_panels(N, shape, lim=None, pos=None):

	# initialize list of panels
	panels = []

	# initialize closed parameter
	closed = False

	# deal with lim/pos not given:
	if lim is None:
		lim = [-1, 1]
	if pos is None:
		pos = 0

	# length of segment
	def dist(x, y):
		return (x**2 + y**2)**(1/2)
	
	# angle of segment
	def ang(x, y):
		return np.arctan2(y, x)

	if shape == 'circle':
		# parametrization
		lmd = np.linspace(0, 2 * np.pi, N, endpoint=False)
		x = np.cos(lmd)
		y = np.sin(lmd)

		# closed shapes can all be panelized easily
		closed = True

	elif shape == 'hline':
		# parametrization
		x = np.linspace(lim[0], lim[1], N)

		for i in range(len(x)-1):
			xm = (x[i] + x[i+1]) / 2
			ym = pos
			s = np.abs(x[i+1] - x[i])
			phi = 0 if x[i+1] >= x[i] else np.pi

			if s != 0: panels.append([xm, ym, phi, s])

		groups = np.array([range(len(panels))])

	elif shape == 'vline':
		# parametrization
		x = np.array([pos] * N)
		y = np.linspace(lim[0], lim[1], N)

		for i in range(len(y)-1):
			xm = pos
			ym = (y[i] + y[i+1]) / 2
			s = np.abs(y[i+1] - y[i])
			phi = np.pi / 2 if y[i+1] >= y[i] else -np.pi / 2

			if s != 0: panels.append([xm, ym, phi, s])

		groups = np.array([range(len(panels))])

	elif shape == 'capacitor':
		# lower plate
		pan1, grp1 = make_panels(N//2, 'hline', lim=[-1, 1], pos=-0.5)
		# upper plate
		pan2, grp2 = make_panels(N//2, 'hline', lim=[-1, 1], pos=0.5)
		
		# fix indexing of second group
		grp2 = grp2 + len(grp1[0])

		# join lists
		panels = np.concatenate((pan1, pan2))
		groups = np.concatenate((grp1, grp2))

	elif shape == 'L':
		# lower plate
		pan1, grp1 = make_panels(N//2, 'hline', lim=[1, 0], pos=0)
		# upper plate
		pan2, grp2 = make_panels(N//2, 'vline', lim=[0, 1], pos=0)

		# fix indexing of second group
		grp2 = grp2 + len(grp1[0])

		# join lists
		panels = np.concatenate((pan1, pan2))
		groups = np.concatenate((grp1, grp2))
	
	else:
		raise Exception(f'Shape {shape} has not been implemented.')

	if closed:
		for i in range(-1, N-1):
			xm = (x[i] + x[i+1]) / 2
			ym = (y[i] + y[i+1]) / 2
			s = dist(x[i+1] - x[i], y[i+1] - y[i])
			phi = ang(x[i+1] - x[i], y[i+1] - y[i])

			if s != 0: panels.append([xm, ym, phi, s])

		groups = np.array([range(len(panels))])

	return np.array(panels), np.array(groups)

# Create matrix G
def make_G(panels):

	#  initialize matrix
	G = np.zeros((len(panels), len(panels)))

	# loop over panels
	for i in range(len(panels)):
		for j in range(len(panels)):
			
			# rotate coordinates into CM frame
			xp = ((panels[j,0] - panels[i,0]) * np.cos(panels[i,2]) -
				(panels[j,1] - panels[i,1]) * np.sin(panels[i,2]))
			yp = ((panels[j,0] - panels[i,0]) * np.sin(panels[i,2]) +
				(panels[j,1] - panels[i,1]) * np.cos(panels[i,2]))

			# calculate G[i,j] from formula
			G[i,j] = G0(xp, yp, panels[i,3])

	return G

# Create the vector u
def make_u(N, groups, pots):
	assert len(groups) == len(pots)

	# initialize vector u
	u = np.zeros(N)

	# loop over the panels
	for i in range(N):

		# lop over the possible boundary pieces
		for n in range(len(pots)):
			if i in groups[n]:
				u[i] = pots[n]

	return u

def solve_profile(N, shape, pots=None, ret_cap=False, u_plot=None):

	# panels are characterized by midpoint xm, ym, angle phi and length s
	panels, groups = make_panels(N, shape)

	if pots == None:
		pots = [1] * len(groups)

	# create matrix G
	G = make_G(panels)

	# create vector u
	u = make_u(len(panels), groups, pots)

	# solve system for charge distribution
	sig = la.solve(G, u, overwrite_a=True, overwrite_b=True)

	# calculate solution at generic point
	def sol(x, y):

		# sum over all panels
		sm = 0
		for i in range(len(panels)):

			# rotate coordinates into CM frame
			xp = ((x - panels[i,0]) * np.cos(panels[i,2]) -
				(y - panels[i,1]) * np.sin(panels[i,2]))
			yp = ((x - panels[i,0]) * np.sin(panels[i,2]) +
				(y - panels[i,1]) * np.cos(panels[i,2]))

			# contibution from formula
			sm += sig[i] * G0(xp, yp, panels[i,3])

		return sm

	if u_plot is not None:
		
		# intervas to plot
		x = np.linspace(u_plot['xlim'][0], u_plot['xlim'][1], u_plot['xnum'])
		y = np.linspace(u_plot['ylim'][0], u_plot['ylim'][1], u_plot['ynum'])

		# create xy mesh
		X, Y = np.meshgrid(x, y)

		# calculate solution at each mesh point
		U = sol(X, Y)

		# Plotting
		fig, ax = plt.subplots(1)
		plt.title(fr'Potential Profile, $N={len(panels)}$')
		plt.xlabel(r'$x$ coordinate')
		plt.ylabel(r'$y$ coordinate')
		ax.set_aspect('equal')

		# plot the boundary
		for group in groups:
			plt.plot(panels[group,0], panels[group,1], color='k')

		# contour plot with levels
		cnt = plt.contourf(X, Y, U, levels=200)
		plt.contour(X, Y, U, colors='k', linestyles='dashed', levels=10)

		clb = plt.colorbar(cnt, shrink=0.7, aspect=9)

		# save the figure if necessary
		if 'savefig' in u_plot: plt.savefig(u_plot['savefig'])
		if 'showfig' in u_plot: plt.show()

	if ret_cap:
		cap = np.sum(sig * panels[:,3]) / 2

		return sig, cap
		
	return sig

N = 100
shape = 'circle'
u_plot = {'xlim': [-2.5, 2.5], 'ylim': [-1.5, 1.5], 'xnum': 200, 'ynum': 200,
	'showfig': True}#, 'savefig': f'images/{shape}_profile_{N}.png'}

sig, cap = solve_profile(N, shape, ret_cap=True, u_plot=u_plot)
panels, groups = make_panels(N, shape)

print(sig)
print(panels)

fig, ax = plt.subplots(1)
plt.title(fr'Charge Density over the circle, $N={len(panels)}$')
plt.xlabel(r'Length along the circle $x$')
plt.ylabel(r'Charge Density $\sigma(x)$')
plt.grid()

for n in range(len(groups)):
	plt.plot(np.linspace(0, 2*np.pi, len(panels)), sig)
plt.axvline(0, color='k', linestyle='dashed')
plt.axvline(2*np.pi, color='k', linestyle='dashed')

plt.savefig(f'images/sigma_x_{shape}_{N}.png')
plt.show()

print(cap)


"""
#------------------------------------------------------------------------------
# Testing speed of implementation
shape = 'circle'

# different n
num_lst = range(4, 100)

# Initizalize list
times = []

for n in num_lst:

	print(f'Calculating for n={n}')

	start = timer()
	solve_profile(n, shape)
	end = timer()
	times.append(end - start)

print('times:', times)

xlabel = r'Number of Panels $\ln(N)$'
ylabel = r'Evaluation Time $\ln(t/[s])$'
title = fr'Evaluation Time for Different Numbers of Panels'
#ylim = [-6, 3.9]
savepath = 'images/times_fit.png'
textbox = {'params': [0], 'symbols': ['slope'], 'coords': [0.6, 0.3], 'error': True}
FitterPlotter('lin', [1.0, 1.0], np.log(num_lst), np.log(times), nstd=1,
	title=title, xlabel=xlabel, ylabel=ylabel, textbox=textbox, savepath=savepath)

#------------------------------------------------------------------------------
"""