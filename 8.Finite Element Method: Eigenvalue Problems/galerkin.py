import numpy as np
import scipy.special as sp
import scipy.linalg as la

import matplotlib.pyplot as plt
from matplotlib import rc

from timeit import default_timer as timer


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# trial functions
def Psi(m, n, xi, phi, ftype='maii'):
	if ftype == 'mfp':
		return xi**(2*m+1) * (1-xi)**(n+1) * np.sin((2*m+1)*phi)
	elif ftype == 'maii':
		return xi**(m+n) * (1-xi) * np.sin(m*phi)

# matrix A
def A(nt, m, n, ftype='maii'):
	if ftype == 'mfp':
		return -np.pi/2 * ((n+1)*(nt+1)*(4*m+3))/(n+nt+4*m+4) * sp.beta(n+nt+1, 4*m+3)
	elif ftype == 'maii':
		return np.pi/2 * ((n*nt)/(2*m+n+nt)-(2*n*nt+n+nt)/(2*m+n+nt+1)+((n+1)*(nt+1))/(2*m+2+n+nt))
		#return np.pi/2 * ((m+n)*(1-m-n)/(2*m-1+n+nt)+2*(m+n)**2/(2*m+n+nt)-(m+n+2*m*n+n**2)/(2*m+1+n+nt)-2*m**2/(2*m+2+n+nt)+m**2/(2*m+3+n+nt))

# matrix B
def B(nt, m, n, ftype='maii'):
	if ftype == 'mfp':
		return 0 # WIP
	elif ftype == 'maii':
		return np.pi/2 * (1/(2*m+2+n+nt)-2/(2*m+3+n+nt)+1/(2*m+4+n+nt))
		#return np.pi/2 * (2/((2*m+1+n+nt)*(2*m+2+n+nt)*(2*m+3+n+nt)))

# do the calculation
def calc(M, N, eigs, to_calc_u=False, to_plot_u=False, xi_num=1000, phi_num=1000):

	# Setup of parameters for calculating u
	if to_calc_u:
		
		# Mesh grid for plotting and calculating u
		# ksi is wighted more towards outer edge
		xi = np.linspace(0, 1, xi_num)
		phi = np.linspace(0, 1, phi_num)**(1/3) * np.pi

		r, theta = np.meshgrid(xi, phi)

		# Initial matrix of values and error
		u = np.zeros((M, eigs, xi_num, phi_num))
		deltaumax_lst = []
		lmd = np.zeros((M, eigs))

	# size of vector set
	m_lst = range(1, M)
	n_lst = range(N)

	# system is decomposable in m, so we solve for every m separately
	for m in m_lst:

		# indicator
		print(f'Solving for m={m} out of M={M} with N={N}')

		# for each m we solve an eigenvalue problem
		# with matrix A
		A_mat = np.fromfunction(lambda nt, n: A(nt, m, n), (N, N))

		# and matrix B
		B_mat = np.fromfunction(lambda nt, n: B(nt, m, n), (N, N))

		# solve the system
		ep, vp = la.eigh(A_mat, B_mat, subset_by_index=[0, eigs-1])

		if to_calc_u:
		
			# vector of matrices for trial functions
			psi = np.fromfunction(lambda n: Psi(m, n, xi[:, None, None],
				phi[:, None]), (N,))

			print(psi.shape)

			# build u
			for i in range(eigs):
				
				# part of solution up
				up = np.sum(psi * vp[:,i].reshape((1,1,-1)), axis=2)
				deltaumax_lst.append(np.max(np.abs(up)))
				u[m,i] += up
		
			lmd[m,:] = ep

	# Plotting velocity profile
	if to_plot_u and to_calc_u:
		fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
		
		ax.set_title(r'Profile of Flow in Semicircular Pipe')
		ax.set_xlim(0, np.pi)

		plt.contour(theta, r, u[2,0].T, 7, colors='k',
			linestyles='dashed', linewidths=1)
		surf = ax.contourf(theta, r, u[2,0].T, 200)

		clb = fig.colorbar(surf, shrink=0.7, aspect=9)
		clb.ax.set_title(r'Velocity $u$')

		plt.savefig('images/flow_profile.png')
		plt.show()

	if to_calc_u:
		return lmd, u, deltaumax_lst

	return None

r"""
#------------------------------------------------------------------------------
# Testing speed of implementation

# Parameter N
N = 50

# different M
M_num = 350
M_lst = range(M_num)

# Initizalize list of times
times_M = []

for M in M_lst:
	start = timer()
	y = calc(M, N)
	end = timer()
	times_M.append(end - start)


# Parameter M
M = 50

# different N
N_num = 350
N_lst = range(N_num)

# Initizalize list of times
times_N = []

for N in N_lst:
	start = timer()
	y = calc(M, N)
	end = timer()
	times_N.append(end - start)


# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Number of Functions $N$ / $M$')
plt.ylabel(r'Evaluation Time $t$ [s]')
plt.title(fr'Evaluation Time for Different Numbers of Functions')
plt.grid()

plt.plot(M_lst, times_M, 'o-', label=r'Constant $N$')
plt.plot(N_lst, times_N, 'o-', label=r'Constant $M$')

plt.legend(loc='best')
plt.savefig('images/times.png')
plt.show()


figure, axes = plt.subplots(1)

plt.xlabel(r'Number of Functions $N$ / $M$')
plt.ylabel(r'Evaluation Time $t$ [s]')
plt.title(fr'Evaluation Time for Different Numbers of Functions')
plt.xscale('log')
plt.yscale('log')
plt.grid()

plt.plot(M_lst[1:], times_M[1:], 'o-', label=r'Constant $N$')
plt.plot(N_lst[1:], times_N[1:], 'o-', label=r'Constant $M$')

plt.legend(loc='best')
plt.savefig('images/times_log.png')
plt.show()

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Testing convergence of implementation

# set density of sampling
xi_num = 500
phi_num = 500

# At constant N it's easy to do since it's already calculated
# Parameter N
N = 30
M = 150
M_lst = range(M)

# calculate errors
_, deltaumax_lst_M, C_lst_M = calc(M, N, to_calc_u=True,
	xi_num=xi_num, phi_num=phi_num)
deltaC_lst_M = np.abs(C_lst_M[1:] - C_lst_M[:-1])


# At constant M we have to work harder
# Parameter M
M = 30

# different N
N_num = 150
N_lst = range(N_num)

# Initizalize lists
deltaumax_lst_N = []
C_lst_N = []

for N in N_lst:
	
	# calculate errors
	_, deltaumax_lst, C_lst = calc(M, N, to_calc_u=True,
		xi_num=xi_num, phi_num=phi_num)

	# append N lists
	deltaumax_lst_N.append(deltaumax_lst[-1])
	C_lst_N.append(C_lst[-1])

deltaC_lst_N = np.abs(np.array(C_lst_N[1:]) - np.array(C_lst_N[:-1]))


# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Number of Functions $N$ / $M$')
plt.ylabel(r'Error Estimate $\Delta C$')
plt.title(fr'Error Estimate for $C$ vs. Number of Functions')
plt.xscale('log')
plt.yscale('log')
plt.grid()

plt.plot(M_lst[1:], deltaC_lst_M, 'o-', label=r'Constant $N$')
plt.plot(N_lst[1:], deltaC_lst_N, 'o-', label=r'Constant $M$')

plt.legend(loc='best')
plt.savefig('images/deltaC_log.png')
plt.show()


figure, axes = plt.subplots(1)

plt.xlabel(r'Number of Functions $N$ / $M$')
plt.ylabel(r'Error Estimate $\mathrm{max}_{\xi, \phi} (\Delta u)$')
plt.title(fr'Error Estimate for $u$ vs. Number of Functions')
plt.xscale('log')
plt.yscale('log')
plt.grid()

plt.plot(M_lst, deltaumax_lst_M, 'o-', label=r'Constant $N$')
plt.plot(N_lst, deltaumax_lst_N, 'o-', label=r'Constant $M$')

plt.legend(loc='best')
plt.savefig('images/deltaumax_log.png')
plt.show()

#------------------------------------------------------------------------------
"""
#------------------------------------------------------------------------------
# Plotting the profile

# Parameters M and N
M = 7
N = 10

# calculate C and plot u 
lmd, _, _ = calc(M, N, 3, to_calc_u=True, to_plot_u=True)

print(np.sqrt(lmd))
print(sorted(np.concatenate(lmd)))

#------------------------------------------------------------------------------
 
