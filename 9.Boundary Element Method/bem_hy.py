import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
from matplotlib import rc
#from FitterPlotter import FitterPlotter

from timeit import default_timer as timer


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# parametrize boundary
def make_panels(N, shape, lim=None, pos=None):

	# initialize list of panels
	panels = []

	# length of segment
	def dist(x, y):
		return (x**2 + y**2)**(1/2)
	
	# angle of segment
	def ang(x, y):
		return np.arctan2(y, x)

	try:
		if shape['name'] == 'line':
		
			if shape['orientation'] == 'h':
				# parametrization
				x = np.linspace(shape['lim'][0], shape['lim'][1], N)
				y = np.array([shape['pos']] * N)

				for i in range(len(x)-1):
					xm = (x[i] + x[i+1]) / 2
					ym = shape['pos']
					s = np.abs(x[i+1] - x[i])
					phi = 0 if x[i+1] >= x[i] else np.pi

					if s != 0: panels.append([xm, ym, phi, s])
		
			elif shape['orientation'] == 'v':
				# parametrization
				x = np.array([shape['pos']] * N)
				y = np.linspace(shape['lim'][0], shape['lim'][1], N)

				for i in range(len(y)-1):
					xm = shape['pos']
					ym = (y[i] + y[i+1]) / 2
					s = np.abs(y[i+1] - y[i])
					phi = np.pi / 2 if y[i+1] >= y[i] else -np.pi / 2

					if s != 0: panels.append([xm, ym, phi, s])

		elif shape['name'] == 'capacitor':
			# lower plate
			pan1 = make_panels(N, {'name': 'line', 'closed': False, 'orientation': 'h',
				'lim': [-1, 1], 'pos': -0.5})
			# upper plate
			pan2 = make_panels(N, {'name': 'line', 'closed': False, 'orientation': 'h',
				'lim': [-1, 1], 'pos': 0.5})
		
			# join lists
			panels = np.concatenate((pan1, pan2))
		
		elif shape['name'] == 'L':
			# lower plate
			pan1 = make_panels(N, {'name': 'line', 'closed': False, 'orientation': 'h',
				'lim': [1, 0], 'pos': 0})
			# upper plate
			pan2 = make_panels(N, {'name': 'line', 'closed': False, 'orientation': 'v',
				'lim': [0, 1], 'pos': 0})

			# join lists
			panels = np.concatenate((pan1, pan2))

		elif shape['name'] == 'ellipse':
			# parametrization
			lmd = np.linspace(0, 2 * np.pi, N)
			x = shape['a'] * np.cos(lmd)
			y = shape['b'] * np.sin(lmd)

		elif shape['name'] == 'fishfin':
			#parametrization
			lmd = np.linspace(0, 2)

	
		else:
			raise Exception(f'Shape {shape} has not been implemented.')

		if shape['closed']:
			for i in range(N-1):
				xm = (x[i] + x[i+1]) / 2
				ym = (y[i] + y[i+1]) / 2
				s = dist(x[i+1] - x[i], y[i+1] - y[i])
				phi = ang(x[i+1] - x[i], y[i+1] - y[i])

				if s != 0: panels.append([xm, ym, phi, s])

	except KeyError:
		raise KeyError(f'Dictionary {shape} does not have required keys.')

	return np.array(panels), np.array([x, y])

# Green's function in CM frame
def V0(x, y, l):
	xm = x - l / 2
	xp = x + l / 2

	if y == 0 and xm == 0 and xp == 0:
		raise Exception('Expession not defined for x=y=l=0.')
	if y == 0 and (xm > 0 or xp < 0):
		return [np.zeros(x.shape) if type(x) is np.ndarray else 0,
		1 / (2 * np.pi) * np.log(np.abs(xp / xm))]
	if y == 0 and xm < 0 and xp > 0:
		return [1/2 * np.ones(x.shape) if type(x) is np.ndarray else 1/2,
		1 / (2 * np.pi) * np.log(np.abs(xp / xm))]
	if xm == 0:
		return [1 / (2 * np.pi) * np.arctan(xp / y),
			1 / (4 * np.pi) * (xp / y)**2]
	if xp == 0:
		return [- 1 / (2 * np.pi) * np.arctan(xm / y),
			- 1 / (4 * np.pi) * (xm / y)**2]

	return [1 / (2 * np.pi) * (np.arctan(xp / y) - np.arctan(xm / y)),
		1 / (4 * np.pi) * np.log((xp**2 + y**2) / (xm**2 + y**2))]
V0 = np.vectorize(V0)

# Create matrix V
def make_V(panels):

	#  initialize matrix
	V = np.zeros((len(panels), len(panels)))

	# loop over panels
	for i in range(len(panels)):
		for j in range(len(panels)):
			
			# rotate coordinates into CM frame
			xp = ((panels[j,0] - panels[i,0]) * np.cos(panels[i,2]) +
				(panels[j,1] - panels[i,1]) * np.sin(panels[i,2]))
			yp = (-(panels[j,0] - panels[i,0]) * np.sin(panels[i,2]) +
				(panels[j,1] - panels[i,1]) * np.cos(panels[i,2])) 

			# calculate vp and vt from formula
			v = V0(xp, yp, panels[i,3])

			# rotate vp/vt into vp in j frame
			V[i,j] = (v[0] * np.cos(panels[i,2] - panels[j,2]) +
				v[1] * np.sin(panels[i,2] - panels[j,2]))

	return V

# Create the vector u
def make_u(panels, u_inf):
		
	# initialize vector u
	u = np.zeros(len(panels))

	# loop over the panels
	for i in range(len(panels)):

		# perpendicular component of velocity
		u[i] =  (u_inf[1] * np.cos(panels[i,2]) -
			u_inf[0] * np.sin(panels[i,2]))

	return u

def solve_profile(N, shape, u_inf, u_plot=None):

	# panels are characterized by midpoint xm, ym, angle phi and length s
	panels, bound = make_panels(N, shape)

	# create matrix V
	V = make_V(panels)

	# create vector u
	u = make_u(panels, u_inf)

	# solve system for charge distribution
	sig = la.solve(V, u, overwrite_a=True, overwrite_b=True)

	print(sig)

	# calculate solution at generic point
	def sol(x, y):

		# sum over all panels
		sm_vx = 0
		sm_vy = 0
		for i in range(len(panels)):

			# rotate coordinates into CM frame
			xp = ((x - panels[i,0]) * np.cos(panels[i,2]) +
				(y - panels[i,1]) * np.sin(panels[i,2]))
			yp = (-(x - panels[i,0]) * np.sin(panels[i,2]) +
				(y - panels[i,1]) * np.cos(panels[i,2]))

			# contibution in CM frame from formula
			v = sig[i] * V0(xp, yp, panels[i,3])

			# rotate vp/vt into LAB frame
			vy = (v[0] * np.cos(panels[i,2]) + v[1] * np.sin(panels[i,2]))
			vx = (v[1] * np.cos(panels[i,2]) - v[0] * np.sin(panels[i,2]))

			# add contributions
			sm_vx += vx
			sm_vy += vy 

		return (sm_vx + u_inf[0], sm_vy + u_inf[1])

	if u_plot is not None:
		
		# intervas to plot
		x = np.linspace(u_plot['xlim'][0], u_plot['xlim'][1], u_plot['xnum'])
		y = np.linspace(u_plot['ylim'][0], u_plot['ylim'][1], u_plot['ynum'])

		# create xy mesh
		X, Y = np.meshgrid(x, y)

		# initialize solution matrices
		VX = np.zeros((len(x), len(y)))
		VY = np.zeros((len(x), len(y)))

		# calculate solution at each mesh point
		# looping is slow, but vectorize is difficult when output is vector
		for i in range(u_plot['xnum']):
			for j in range(u_plot['ynum']):
				v = sol(X[i,j], Y[i,j])
				if v[0]**2+v[1]**2 < 1.9**2:
					VX[i,j] = v[0]
					VY[i,j] = v[1]

		v = []
		print(bound)
		for i in range(len(bound[0])):
			v.append(sol(1.00000001 * bound[0,i], 1.00000001 * bound[1,i]))
		v = np.array(v)
		print('vperp:', v[:,0])
		print('vpara:', v[:,1])

		plt.plot((v[:,0]**2+v[:,1]**2)**(1/2))
		plt.show()


		# Plotting
		fig, ax = plt.subplots(1)
		plt.title(fr'Flow Profile, $N={len(panels)}$')
		plt.xlabel(r'$x$ coordinate')
		plt.ylabel(r'$y$ coordinate')
		ax.set_aspect('equal')

		# plot the boundary
		plt.plot(bound[0], bound[1], color='k')

		vct = plt.quiver(X, Y, VX, VY, (VX**2 + VY**2)**(1/2))

		# contour plot with levels
		#cnt = plt.contourf(X, Y, U, levels=200)
		#plt.contour(X, Y, U, colors='k', linestyles='dashed', levels=10)

		clb = plt.colorbar(vct, shrink=0.7, aspect=9)

		# save the figure if necessary
		if 'savefig' in u_plot: plt.savefig(u_plot['savefig'])
		if 'showfig' in u_plot: plt.show()

	return sig

#u_inf = [1/np.sqrt(2), 1/np.sqrt(2)]
u_inf = [0, 1]


u_plot = {'xlim': [-3.5, 3.5], 'ylim': [-2.5, 2.5], 'xnum': 20, 'ynum': 20,
	'showfig': True}#, 'savefig': 'images/flow_ellipse_1.png'}

shape = {'name': 'ellipse', 'closed': True, 'a': 1.5, 'b': 1}
#shape = {'name': 'capacitor', 'closed': False}

#sig = solve_profile(100, shape, u_inf, u_plot=u_plot)

t = np.linspace(0, 2* np.pi, 500)
x = np.array([0, 0.06346651825, 0.1269330365, 0.1903995548, 0.253866073, 0.3173325913, 0.3807991095, 0.4442656278, 0.507732146, 0.5711986643, 0.6346651825, 0.6981317008, 0.7615982191, 0.8250647373, 0.8885312556, 0.9519977738, 1.015464292, 1.07893081, 1.142397329, 1.205863847, 1.269330365, 1.332796883, 1.396263402, 1.45972992, 1.523196438, 1.586662956, 1.650129475, 1.713595993, 1.777062511, 1.840529029, 1.903995548, 1.967462066, 2.030928584, 2.094395102, 2.157861621, 2.221328139, 2.284794657, 2.348261175, 2.411727694, 2.475194212, 2.53866073, 2.602127248, 2.665593767, 2.729060285, 2.792526803, 2.855993321, 2.91945984, 2.982926358, 3.046392876, 3.109859394, 3.173325913, 3.236792431, 3.300258949, 3.363725467, 3.427191986, 3.490658504, 3.554125022, 3.61759154, 3.681058059, 3.744524577, 3.807991095, 3.871457614, 3.934924132, 3.99839065, 4.061857168, 4.125323687, 4.188790205, 4.252256723, 4.315723241, 4.37918976, 4.442656278, 4.506122796, 4.569589314, 4.633055833, 4.696522351, 4.759988869, 4.823455387, 4.886921906, 4.950388424, 5.013854942, 5.07732146, 5.140787979, 5.204254497, 5.267721015, 5.331187533, 5.394654052, 5.45812057, 5.521587088, 5.585053606, 5.648520125, 5.711986643, 5.775453161, 5.838919679, 5.902386198, 5.965852716, 6.029319234, 6.092785752, 6.156252271, 6.219718789, 6.283185307])
y = np.array([0.0239145605, 0.052077727, 0.134010949, 0.262558179, 0.427323578, 0.6164370185, 0.8182321415, 1.02248754, 1.22110381, 1.408275407, 1.580311002, 1.735268442, 1.872534052, 1.99242744, 2.095870504, 2.184130773, 2.25863373, 2.320832089, 2.372118467, 2.413769719, 2.446913225, 2.472507694, 2.491333343, 2.503987621, 2.51088393, 2.512251728, 2.508136879, 2.498401749, 2.482724972, 2.460601065, 2.431340914, 2.394074244, 2.34775628, 2.291181766, 2.223010548, 2.141811313, 2.046131635, 1.934605406, 1.806110386, 1.659989097, 1.496342671, 1.316396709, 1.122916362, 0.9206124095, 0.716432773, 0.5195879515, 0.3411421065, 0.193051604, 0.086675224, 0.0309963525, 0.0309963525, 0.086675224, 0.193051604, 0.3411421065, 0.51958794, 0.716432773, 0.9206124095, 1.122916362, 1.316396709, 1.496342671, 1.659989097, 1.806110386, 1.934605406, 2.046131635, 2.141811313, 2.223010548, 2.291181766, 2.34775628, 2.394074232, 2.431340914, 2.460601065, 2.482724972, 2.498401749, 2.508136879, 2.512251728, 2.51088393, 2.503987621, 2.491333343, 2.472507682, 2.446913225, 2.413769719, 2.372118467, 2.320832089, 2.25863373, 2.184130761, 2.095870504, 1.99242744, 1.872534052, 1.735268442, 1.580311002, 1.408275407, 1.22110381, 1.02248754, 0.8182321415, 0.6164370185, 0.427323578, 0.262558179, 0.134010949, 0.052077727, 0.0239145605])


fig, ax = plt.subplots(1)
plt.title(fr'Parallel velocity over the ellipse, $N=100$')
plt.xlabel(r'Angle along the ellipse $t$')
plt.ylabel(r'Parallel velocity $|v_{\parallel}|$')
plt.grid()

plt.plot(t, np.abs(2.5*np.sin(t) / (2.25 - 1.15*(np.sin(t))**2)**(1/2)), label='Analytical')
plt.plot(x, 1.09 * y/1.15, 'o', label='Calculated')

plt.legend(loc='upper right')
plt.savefig(f'images/v_para_t.png')
plt.show()
