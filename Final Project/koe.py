import diffeq_2 as de

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
rc('text', usetex=True)


def spread_test(n, N, T, mode='raw'):

	spread = []
	
	for _ in range(n):
		omega = np.random.rand(N)
		omega = omega - np.average(omega)
		omega = omega * 0.1

		x0 = np.random.rand(N) * (2 * np.pi)

		_, x = solve_kuramoto(N, omega, T, x0=x0)

		delta = np.max(np.mod(x[-1], 2*np.pi)) - np.min(np.mod(x[-1], 2*np.pi))
		if delta < np.pi:
			spread.append(delta)
		else:
			delta = np.max(np.mod(x[-1] + np.pi, 2*np.pi)) - np.min(np.mod(x[-1] + np.pi, 2*np.pi))
			spread.append(delta)

	if mode == 'cummean':

		cummean = [spread[0]]
		for i in range(1,n):
			cummean.append(cummean[i-1] * (i-1)/i + spread[i]/i)
		
		return cummean
	
	else:
		return spread

def hist_double_sums(n, N, prnt='False'):
	
	a = []
	for t in range(n):

		x0 = np.random.rand(N) * (2 * np.pi)

		s = 0
		for i in range(N):
			for j in range(N):
				s += (x0[i] - x0[j]) * np.sin(x0[i] - x0[j])
	
		if prnt: print('Iteration:', t, '\t', 's:', s / N**2)

		a.append(s / N**2)

	print(a)

	_ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram

	plt.title("Histogram with 'auto' bins")
	plt.show()

# check the eigenvalues of matrix
def arb_mat_check(N):

	x = np.random.rand(N, N)
	x = x + x.T - 1
	y = np.zeros((N, N))
	for i in range(N): y[i,i] = np.sum(x, axis=0)[i]
	z = x - y

	print(np.linalg.det(x - y))
	print(np.all(np.linalg.eigvals(z) <= 10**(-10)))
	print(np.linalg.eigvals(z))
	print(np.sum(np.linalg.eigvals(z) <= 10**(-10))/N)

def solve_std():

	def g(x, t):
		return - np.sqrt(x) + 2 * t + k

	# tolerance and step for integrator
	tol = 10**(-8)
	hmin = 10**(-10)
	hmax = 0.5
	
	k = 0
	t1, x1 = de.rkf(g, 0, 1, np.pi/np.sqrt(3), tol, hmax, hmin)

	k = 1
	t2, x2 = de.rkf(g, 0, 1, np.pi/np.sqrt(3), tol, hmax, hmin)

	k = -1
	t3, x3 = de.rkf(g, 0, 1, np.pi/np.sqrt(3), tol, hmax, hmin)

	plt.plot(t1, x1)
	plt.plot(t2, x2)
	plt.plot(t3, x3)
	plt.grid()
	plt.show()

# Gaussian distribution
def gauss(x, mu=0, sigma=1):
	return 1/((2*np.pi)**(1/2) * sigma) * np.exp(-(x - mu)**2 / (2*sigma**2))

# Lorentzian distribution
def lorentz(x, x0=0, gamma=1):
	return 1/(np.pi*gamma) * 1/(1 + ((x - x0) / gamma)**2)

# Unifrom distribution
def uniform(x, x0=0, sgm=1):
	return (np.heaviside(x - x0 + sgm, 1/2) -
		np.heaviside(x - x0 - sgm, 1/2))/(2 * sgm)

# generate omega according to some distribution
def make_omega(N, mode, spread):

	# random number generator
	rng = np.random.default_rng()

	# different distributions
	if mode == 'uniform':
		omega = rng.uniform(-spread, spread, N)
	elif mode == 'normal':
		omega = rng.normal(0, spread, N)
	elif mode == 'lorentz':
		omega = rng.standard_cauchy(N) * spread / 100
	
	# make sure average is 0
	omega = omega - np.average(omega)

	return omega

# Solve the set of differential equations
def solve_kuramoto(N, omega, T, x0=None, k=1):
	
	# make sure omega is the correct size
	assert len(omega) == N

	# use x0 picked from a uniform distribution
	if x0 is None:
		x0 = np.random.rand(N) * (2 * np.pi)
		x0 = x0 - np.average(x0)
	
	# tolerance and step for integrator
	tol = 10**(-8)
	hmin = 10**(-10)
	hmax = 0.5
	
	# differential equation in required form
	def f(x, t):
		return omega + k / N * np.sum(np.sin(x[:,None] - x), axis=0)

	# solve differential equation 
	x = de.rkf(f, 0, T, x0, tol, hmax, hmin)

	return x

# measure spread of stationary state
def spread_correlation(a, b, n, N, mode='all', speedup=False):
	"""Measures the spread in the final stationary state (if it is achieved)
	as a function of the spread in the distribution of omega. Here spread is
	defined as the difference between the maximal and minimal value.
	"""

	# make sure a is smaller than b
	assert a <= b
	
	# initialize lists for storing results
	spread_omega = {'uniform': [], 'normal': [], 'lorentz': []}
	spread_theta = {'uniform': [], 'normal': [], 'lorentz': []}

	# vector of approximate spreads
	init_spread = np.linspace(a, b, n)

	for sgm in init_spread:

		# indicator
		print(f'Computing for sgm = {sgm}.')
		
		# generate omega list
		omega = {}
		if mode == 'all':
			for md in ['uniform', 'normal', 'lorentz']:
				omega[md] = make_omega(N, md, sgm / 2)
		else:
			omega[mode] = make_omega(N, mode, sgm / 2)

		for md in omega.keys():

			# speed-up using analytic result
			if speedup:
				if np.any(np.abs(omega[md]) > (N - 1)/N):
					spread_omega[md].append(np.nan)
					spread_theta[md].append(np.nan)
					continue
			
			# calculate spread of omega
			spread_omega[md].append(np.max(omega[md]) - np.min(omega[md]))
	
			# solve differential equation
			t, x = solve_kuramoto(N, omega[md], 50)

			# check if we have convergence
			xdot = omega[md] + 1 / N * np.sum(np.sin(x[-1][:,None] - x[-1]), axis=0)
			if not np.allclose(xdot, 0):
				spread_theta[md].append(np.nan)
				continue

			#if spread_omega[-1] > 1.45:
			#	animate(x, mode='spin')

			# calculate spread; take care of mod
			delta = (np.max(np.mod(x[-1], 2*np.pi)) -
				np.min(np.mod(x[-1], 2*np.pi)))
			Delta = (np.max(np.mod(x[-1] + np.pi, 2*np.pi)) -
				np.min(np.mod(x[-1] + np.pi, 2*np.pi)))
			if delta > Delta: delta = Delta
			
			if delta > 3:
				# what the hell is this
				plt.plot(t, x)
				plt.grid()
				plt.show()
				spread_theta[md].append(np.nan)
			else:
				spread_theta[md].append(delta)

	# plotting
	fig, ax = plt.subplots(1)

	ax.set_title(fr'Correlation of Spreads, $\omega$ {mode}, $N={N}$')
	ax.set_ylabel(r'Stationary Angle Spread $\Delta \theta_s$')
	ax.set_xlabel(r'Natural Frequency Spread $\Delta \omega$')
	ax.grid()
	
	for md in spread_omega.keys():
		if len(spread_omega[md]) != 0:
			ax.plot(spread_omega[md], spread_theta[md], 'o', label=md)
	
	omega_dummy = np.linspace(a, b, 1000)
	ax.plot(omega_dummy, omega_dummy, 'k-', label='Linear Result')

	ax.legend()
	fig.savefig(f'images/spreads_{mode}_{a}_{b}_{n}_{N}.png')
	plt.show()

	return spread_omega, spread_theta

# animate motion of oscillators
def animate(x, equal=True, savepath=None, mode='color'):

	# shape of solution
	T, N = x.shape

	# create figure and adding axes
	fig, ax = plt.subplots(1)

	ax.set_title(fr'Kuramoto Oscillators, $N={N}$')
	ax.set_xlim(-1.1, 1.1)
	ax.set_ylim(-1.1, 1.1)
	if equal == True: ax.set_aspect('equal')

	# draw a guiding circle
	phi = np.linspace(0, 2*np.pi, 1000)
	ax.plot(np.cos(phi), np.sin(phi), 'k-', linewidth=1)

	# color mode WIP
	if mode == 'color':
		x = np.mod(x, 2*np.pi)
		phi = np.linspace(0, 2*np.pi, N, endpoint=False)

		sct = ax.scatter(np.cos(phi), np.sin(phi), s=100, c=x[0])
	
	# rotational mode
	elif mode == 'spin':
		color = np.linspace(0, 1, N)
		sct = ax.scatter(np.cos(x[0]), np.sin(x[0]), s=70, c=color)

	# update animation frame
	def update_anim(f):
		
		if mode == 'color':
			sct.set_array(x[f])

		elif mode == 'spin':
			sct.set_offsets(np.array([np.cos(x[f]), np.sin(x[f])]).T)
		
		return sct,

	# animate
	anim = FuncAnimation(fig, update_anim, frames=range(T),
		interval=20, blit=True, save_count=50)

	# save animation and show
	if savepath is not None: anim.save('anims/' + savepath, dpi=150,
		fps=30, writer='ffmpeg')
	plt.show()

	return fig, ax

# plot the variance and covaraince
def var_cov_plot(N, omega, t, x, dist, sgm, mode='normal'):

	# take car of different modes separately
	if mode == 'normal':

		# y-axis labels
		ylabel1 = r'Variance $\langle \theta^2 \rangle$'
		ylabel2 = r'Covariance $\langle \theta \omega \rangle$'
		
		# calculate variance and covariance
		var = np.average(x**2, axis=1)
		cov = np.average(x * omega, axis=1)
	elif mode == 'mod':

		# y-axis labels
		ylabel1 = r'Variance $\langle \left(\mathrm{mod} (\theta, 2\pi)\right)^2 \rangle$'
		ylabel2 = r'Covariance $\langle \mathrm{mod}(\theta, 2\pi) \omega \rangle$'
		
		# take care of mod spiiting angles
		xp = np.mod(x, 2*np.pi)
		if np.max(xp) - np.min(xp) > 1.85 * np.pi:
			xp = np.mod(x + np.pi, 2*np.pi)
		
		# calculate variance and covariance
		var = np.average(xp**2, axis=1)
		cov = np.average(xp * omega, axis=1)

	# create figure and axis 1
	fig, ax1 = plt.subplots()

	# add title and labels
	ax1.set_title(fr'Variance and Covariance over Time, $N={N}$')
	ax1.set_ylabel(ylabel1, color='tab:blue')
	ax1.set_xlabel(r'Time $t$')
	ax1.tick_params(axis='y', labelcolor='tab:blue')
	ax1.grid()
	
	# create axis 2 with shared x
	ax2 = ax1.twinx()

	# add labels
	ax2.set_ylabel(ylabel2, color='tab:orange')  # we already handled the x-label with ax1
	ax2.tick_params(axis='y', labelcolor='tab:orange')

	# plot variance and covariance
	ax1.plot(t, var, color='tab:blue')
	ax2.plot(t, cov, color='tab:orange')
	
	# save and show
	#fig.tight_layout()  # otherwise the right y-label is slightly clipped
	fig.savefig(f'images/varcov_{mode}_{N}_{dist}_{sgm}.png')
	plt.show()

# plot the solution of the problem
def sol_plot(t, x, dist, sgm, mode='normal'):

	# create figure and axes
	fig, ax = plt.subplots()

	# handles mode specific parameters
	if mode == 'normal':
		ylabel = r'Phase Angle $\theta(t)$'
		data = x
	elif mode == 'mod':
		ylabel = r'Phase Angle $\mathrm{mod}(\theta(t), 2\pi)$'
		
		# handles mod splitting data
		data = np.mod(x, 2*np.pi)
		if np.max(data) - np.min(data) >= 1.85 * np.pi:
			data = np.mod(x + np.pi, 2 * np.pi)

	# add title and labels
	ax.set_title(fr'Solution of Differential Equation, $N={N}$')
	ax.set_ylabel(ylabel)
	ax.set_xlabel(r'Time $t$')
	ax.grid()

	# plot the solution
	ax.plot(t, data)
	
	# save and show
	fig.savefig(f'images/solution_{mode}_{N}_{dist}_{sgm}.png')
	plt.show()

# histogram of final positions
def hist_final(omega, xf, dist, sgm):

	# number of samples
	N = len(omega)

	# standardize data and fix mod splitting
	data = np.mod(xf, 2*np.pi)
	if np.max(data) - np.min(data) > 1.85 * np.pi:
		data = np.mod(xf + np.pi, 2*np.pi)
	data = data - np.average(data)

	# create figure and axes
	fig, ax = plt.subplots()
	
	# add title and labels
	ax.set_title(fr'Histogram of Steady State Positions, $N={N}$')
	ax.set_ylabel('Number of Occurences')
	ax.set_xlabel(r'Normalized Final Position $\mathrm{mod}(\theta_f, 2\pi)$')
	ax.grid()

	# draw histogram and probability
	#option 1
	_, bins, _ = ax.hist([data, omega], bins='auto', label=[r'$\theta_f$ Data', r'$\omega$ Data'])
	
	#option 2
	#_, bins, _ = ax.hist(data, bins='auto', alpha=0.5, label=r'$\theta_f$ Data')
	#ax.hist(omega, bins=bins, alpha=0.5, label=r'$\omega$ Data')

	x_dummy = np.linspace(1.1*np.min(data), 1.1*np.max(data), 1000)
	if dist == 'uniform':
		ax.plot(x_dummy, 0.2*N/len(bins) * uniform(x_dummy, sgm=sgm), 'k-',
			label=r'$\omega$ Distribution')
	elif dist == 'normal':
		ax.plot(x_dummy, N / (sgm * len(bins)) * gauss(x_dummy, sigma=sgm), 'k-',
			label=r'$\omega$ Distribution')
	elif dist == 'lorentz':
		ax.plot(x_dummy, N / len(bins) * lorentz(x_dummy, gamma=sgm), 'k-',
			label=r'$\omega$ Distribution')

	# save and show
	ax.legend()
	fig.savefig(f'images/histxf_{N}_{dist}_{sgm}.png')
	plt.show()

def conv_theta0_test(omega, n):
	""" Does convergence only depend on the choice of omega (which is
	suggested by the case study on N=3) or does it also depend on the initial
	conditions theta_0?
	"""

	N = len(omega)

	yes = 0
	no = 0
	x0s = []

	for i in range(n):

		# indicator
		print(f'Iteration {i}/{n}: ', end='')

		# solve differential equation
		_, x = solve_kuramoto(N, omega, 80)

		x0s.append(x[0])

		# check if we have convergence
		xdot = omega + 1 / N * np.sum(np.sin(x[-1][:,None] - x[-1]), axis=0)
		if np.allclose(xdot, 0):
			yes += 1
			print('yes')
		else:
			no += 1
			print('no')
			#plt.plot(t, x)
			#plt.grid()
			#plt.show()

	print('yes:', yes)
	print('no:', no)

	return yes, no
			


# Number of oscillators
N = 1000
print(f'Calculating with N = {N}.')

# generate omega
sgm = 0.1
dist = 'uniform'
omega = make_omega(N, dist, sgm)

# check if omega are too large
print('|omega| >= (N - 1)/N:', np.any(np.abs(omega) >= (N - 1) / N))
print('omega_min:', min(omega))
print('omega_max:', max(omega))

if np.any(np.abs(omega) >= (N - 1) / N): raise SystemExit(0)

#yes, no = conv_theta0_test(omega, 100)

#x0 = make_omega(N, dist, 0)

#omega = np.array([-0.086, 0.052, -0.052 + 0.086])

# solve differential equation
t, x = solve_kuramoto(N, omega, 100)

# check if we have convergence
xdot = omega + 1 / N * np.sum(np.sin(x[-1][:,None] - x[-1]), axis=0)
if np.allclose(xdot, 0):
	print('Converged: Yes')
	print('max derivative:', np.max(np.abs(xdot)))
else:
	print('Converged: No')
	print('max derivative:', np.max(np.abs(xdot)))

hist_final(omega, x[-1], dist, sgm)

# plot solution
sol_plot(t, x, 'wtf', sgm, mode='normal')
#sol_plot(t, [- 1 / N * np.sum(np.sin(x[i][:,None] - x[i]), axis=0) for i in range(len(t))], 'spec_coup', sgm, mode='normal')

# plot solution mod 2pi
#sol_plot(t, x, dist, sgm, mode='mod')

# plot variance and covariance
#var_cov_plot(N, omega, t, x, dist, sgm)

# plot variance and covariance mod 2pi
#var_cov_plot(N, omega, t, x, dist, sgm, mode='mod')

# create animation of the solution
#animate(x, mode='spin', savepath=f'sol_{N}_{dist}_{sgm}.mp4')



# example of non convergence for for N=3
#omega = np.array([0.51, 0.145, - 0.51 - 0.145])
#t,x = solve_kuramoto(3, omega, 80)

#for i in range(N): plt.axhline(omega[i] + np.average(x0))
#plt.axhline(1/2 * np.arcsin(2*omega[0]))
#plt.axhline(-np.pi/2 - 1/2 * np.arcsin(2*omega[0]), linestyle='--')
#plt.axhline(np.mod(1/2 * np.arcsin(2*omega[0]), 2*np.pi)) 
#plt.axhline(np.mod(-np.pi/2 - 1/2 * np.arcsin(2*omega[0]), 2*np.pi), linestyle='--')

#spread = spread_test(200, 50, 25, mode='cummean')
#plt.plot(spread)
#plt.grid()
#plt.show()

#spread_correlation(0, 1.6, 500, 200, mode='all', speedup=True)