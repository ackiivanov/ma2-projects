import scipy.integrate as itg
import scipy.linalg as la
#import bvp_lib as bvp
#import diffeq_2 as de

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt

from memory_profiler import profile
from timeit import default_timer as timer

# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
rc('text', usetex=True)


def pwr_iteration(A, N, z=None, tol=10**(-5), max_itr=7000, saveall=False):

	n, _ = A.shape
	itr = 0
	eigvals = []
	eigvecs = []

	if saveall:
		alleigs = []
		for i in range(N): alleigs.append([])

	if z is None:
		z = np.random.rand(A.shape[1], 1)

	def mult(A, z):
		s = np.zeros((n, 1))
		for i in range(len(eigvals)):
			s += eigvals[i] * (eigvecs[i].T @ z).item() * eigvecs[i]

		return A @ z - s

	def rayl_n(x, A):
		return x.T @ mult(A, x) / (x.T @ x)

	def one_eig_iter(A, z):
		itr = 0

		while la.norm(mult(A, z) - rayl_n(z, A) * z) > tol and itr < max_itr:
			if saveall:
				alleigs[len(eigvals)].append(rayl_n(z, A).item())

			y = mult(A, z)
			z = y / la.norm(y)

			itr += 1

		err = la.norm(mult(A, z) - rayl_n(z, A) * z)
		if err > tol:
			print('You reached the maximal number of iterations before')
			print('reaching the desired tolerance. Your currect error is')
			print(err)
		print('Iterations needed:', itr)

		if saveall: alleigs[len(eigvals)] = np.array(alleigs[len(eigvals)])
		eigvals.append(rayl_n(z, A).item())
		eigvecs.append(z / la.norm(z))

	while len(eigvals) < N:
		print(f'Iterating for eigenvalue number {len(eigvals)}')
		one_eig_iter(A, z)

	if saveall : return alleigs, eigvecs
	else: return np.array(eigvals), np.array(eigvecs)

def jacobi(A, tol=10**(-5), max_itr=500, conv=False):

	def offset(A):
		s = 0
		n, m = A.shape
		for i in range(n):
			for j in range(m):
				if i != j: s += A[i,j]**2

		return np.sqrt(s)

	n, _ = A.shape
	#J = np.eye(n)
	if conv: offall = []

	itr = 0
	off = offset(A)

	while off > tol and itr < max_itr:

		print(f'Currently on iteration number {itr}')
		print(f'Offset is {off}')

		if conv: offall.append(off)

		J = np.eye(n)
		
		for i in range(n):
			for j in range(i):
				if np.abs(A[i,j]) > tol/n**2:
					if A[i,i] != A[j,j]:
						tau = (A[i,i] - A[j,j]) / (2 * A[i,j])
						t = np.sign(tau) / (np.abs(tau) + np.sqrt(1 + tau**2))
						c = 1 / np.sqrt(1 + t**2)
						s = c * t
					else:
						c = 1 / np.sqrt(2)
						s = c

					R = np.eye(n)
					R[i,i] = c
					R[j,j] = c
					R[i,j] = -s
					R[j,i] = s

					A = R.T @ A @ R
					J = J @ R

		off = offset(A)
		itr += 1

	if conv: offall.append(off)
	if off > tol:
		print('You reached the maximal number of iterations before')
		print('reaching the desired tolerance. Your currect offset is')
		print(off)
	print('Iterations needed:', itr)

	if conv: return A, J, offall
	return A, J



"""
N = 100
M = np.random.rand(N, N)
M = M @ M.T
print(M)

eigvals, eigvecs = jacobi(M)
#eigvalsp, eigvecsp = la.eigh(M)

eigvals = np.sort(np.diagonal(eigvals))
print('homebrew:', eigvals)
#print('built-in:', eigvalsp)

#for i in range(N):
#	print(np.abs(eigvals[i] - eigvalsp[i]))

#print(eigvecs)
"""

def mat_to_vec(U):
	N, _ = U.shape

	u = np.zeros(N**2)

	for i in range(N):
		for j in range(N):
			p = i * N + j
			u[p] = U[i,j]

	return u

def vec_to_mat(u):
	n = len(u)

	N = int(np.sqrt(n))
	U = np.zeros((N, N))

	for p in range(n):
		j = p % N
		i = p // N
		U[i,j] = u[p]

	return U

def make_A(N):
	A = np.zeros((N**2, N**2))

	for i in range(N**2):
		A[i,i] = 4

		if i + 1 < N**2 and (i + 1) % N != 0:
			A[i,i+1] = -1
		if i - 1 < N**2 and (i - 1) % N != N-1:
			A[i,i-1] = -1

		if i + N < N**2:
			A[i,i+N] = -1
		if i - N >= 0:
			A[i,i-N] = -1

	return A

def make_Ap(N, shift=10**(-0)):
	A = np.zeros((N**2, N**2))

	for i in range(N**2):
		A[i,i] = 1 + N**2 / (shift + i//N)**2

		if i + 1 < N**2 and (i + 1) % N != 0:
			A[i,i+1] = - N**2 / (shift + i//N)**2
		if i - 1 < N**2 and (i - 1) % N != N-1:
			A[i,i-1] = - N**2 / (shift + i//N)**2

		if i + N < N**2:
			A[i,i+N] = -1 - 1 / (2 * shift + 2 * (i // N))
		if i - N >= 0:
			A[i,i-N] = -1 + 1 / (2 * shift + 2 * (i // N))

	return A


eigvals1, eigvecs = la.eig(make_Ap(50, shift=10**(-10)))
print(np.imag(eigvals1))
eigvals1 = 100 * np.sort(np.real(eigvals1))
eigvals1 = eigvals1[eigvals1 > 0]
print(eigvals1)

eigvals2, eigvecs = la.eig(make_Ap(50, shift=10**(-12)))
eigvals2 = 100 * np.sort(np.real(eigvals2))
eigvals2 = eigvals2[eigvals2 > 0]
print(eigvals2[:10])

plt.yscale('log')
plt.plot(np.abs(eigvals2 - eigvals1))
plt.plot(eigvals1)

plt.show()

def make_B(N, shape, weight):
	B = np.zeros((N**2, N**2))

	for i in range(N):
		for j in range(N):
			if mask(i, j, N, shape): B[i*N+j,i*N+j] = 1
			else: B[i*N+j,i*N+j] = weight

	return B

def mask(i, j, N, shape):
	x = i / N
	y = j / N

	if shape == 'F':
		cond1 = x < 1/4 and y > 1/2
		cond2 = x < 1/4 and y < 1/4
		cond3 = x > 1/2 and y < 1/4
		cond4 = x > 3/4 and 1/4 <= y < 1/2
		cond5 = x > 1/2 and 1/2 <= y < 3/4

		return not (cond1 or cond2 or cond3 or cond4 or cond5)

	if shape == 'square':
		return True
	
	elif shape == 'circle':
		cond1 = ((x - 1/2)**2 + (y - 1/2)**2 <= 1/4)

		return cond1

	elif shape == 'circlebar':
		cond1 = ((x - 1/2)**2 + (y - 1/2)**2 <= 1/4)
		cond2 = x - 1/2 > 0 and np.abs(y - 1/2) < 0.05

		return cond1 and (not cond2)

	elif shape == 'halfcircle':
		cond1 = ((x - 1/2)**2 + (y - 1/2)**2 <= 1/4)
		cond2 = y >= 1/2

		return cond1 and cond2
	
	elif shape == 'birdhouse':
		cond1 = (x < 1/3 and y < 1/3)
		cond2 = (x > 2/3 and y < 1/3)
		cond3 = (y > x + 2/3)
		cond4 = (y > -x + 4/3 and y > 2/3)

		return not (cond1 or cond2 or cond3 or cond4)

	elif shape == 'astroid':
		cond1 = (np.abs(x - 1/2)**(2/3) + np.abs(y - 1/2)**(2/3) <= (1/2)**(2/3))

		return cond1
	else:
		print(f'The given shape "{shape}" is not implemented.')
		raise Exception


def anal(m, n, N):
	x = np.linspace(0, 1, N)

	u = np.sin(np.pi * n * x).reshape((N, 1)) @ np.sin(np.pi * m * x).reshape((1, N))

	s = la.norm(u, ord='fro')

	return u / s

def draw_profile(u, i, N, shape, anal=False):
	fig, ax = plt.subplots(1)

	nm = f'calculated, $N={N}$' if anal == False else 'real'

	ax.set_title(fr'Eigenfunction {i} Profile, ' + nm)
	ax.set_xlabel(r'$x$ coordinate')
	ax.set_ylabel(r'$y$ coordinate')

	if anal == False:
		im = ax.matshow(vec_to_mat(u))
	else:
		im = ax.matshow(u)

	if shape == 'F':
		i_lst = range(0, N)
		j_lst = range(0, N)
		I_lst, J_lst = np.meshgrid(i_lst, j_lst)
		plt.contour(I_lst, J_lst, [[mask(i, j, N, shape) for j in j_lst] for i in i_lst],
			colors='k', linestyles='dashed', linewidths=1, levels=1)

	clb = plt.colorbar(im)
	clb.ax.set_title(r'Amp. $A$')


"""
N = 70
A = make_A(N)
Ainv = la.inv(A)
B = make_B(N, 'F', 100)

#plt.matshow(B)
#plt.show()


eigvals, eigvecs = la.eigh(A)
#eigvals, eigvecs = pwr_iteration(Ainv, 10)

eigs = []
for i in range(1, N+1):
	for j in range(1, N+1):
		eigs.append(np.pi**2 * (i**2 + j**2))


plt.plot(N**2 * eigvals)
#plt.plot(sorted(eigs))
plt.plot()
plt.show()


# be careful of :,i vs i
for i in range(10):
	draw_profile(eigvecs[:,i], i+1, N, 'square')
	plt.savefig(f'images/eigenvector_{i+1}.png')
	plt.show()



plt.matshow(anal(3, 3, 50))
plt.colorbar()
plt.show()
"""

"""
N = 90
A = make_A(N)
#B = make_B(N, 'F', 100)

eigvals, eigvecs = la.eigh(A)

# be careful of :,i vs i
qns = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1), (3, 3)]
for i in range(11):
	draw_profile(eigvecs[:,i], i+1, N, 'square')
	plt.savefig(f'images/eigenvector_{i+1}.png')
	plt.show()

	draw_profile(anal(qns[i][0], qns[i][1], 300), i+1, N, 'square', anal=True)
	plt.savefig(f'images/eigenvector_{i+1}_anal.png')
	plt.show()
"""

"""
# Actual calculation
N = 100
A = make_A(N)
B = make_B(N, 'F', 100)

eigvals, eigvecs = la.eigh(A, B)

# be careful of :,i vs i
qns = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1), (3, 3)]
for i in range(11):
	draw_profile(eigvecs[:,i], i+1, N, 'F')
	plt.savefig(f'images/Feigenvector_{i+1}.png')
	plt.show()
"""

"""
# different rho heavy
N = 70
A = make_A(N)


# Plotting
fig, ax = plt.subplots(1)

ax.set_title(fr'Calculated Eigenvalues for different $\tilde{{\rho}}$, $N={N}$')
ax.set_xlabel(r'Eigenvalue number $n$')
ax.set_ylabel(r'Eigenvalue $k^2$')
ax.set_ylim(-2000, 60000)
ax.grid()

axin = fig.add_axes([0.18, 0.53, 0.38, 0.30])
axin.set_title('First Few')
axin.grid()

weight_lst = [0.2, 0.5, 1, 2, 4, 10]
for weight in weight_lst:

	print(f'Solving for weight: {weight}')
	
	B = make_B(N, 'F', weight)

	eigvals, eigvecs = la.eigh(A, B)
	ax.plot(N**2 * eigvals, label=fr'$\tilde{{\rho}}={weight}$')
	axin.plot(N**2 * eigvals[:10])
	print(N**2 * eigvals[:10])

ax.legend(loc='lower right')
plt.savefig('images/diff_rho_alt.png')
plt.show()
"""

"""
# spectrum
N = 70
A = make_A(N)

eigvals, eigvecs = la.eigh(A)

eigs = []
for i in range(1, N+1):
	for j in range(1, N+1):
		eigs.append(np.pi**2 * (i**2 + j**2))
eigs = np.sort(eigs)

# Plotting
fig, ax = plt.subplots(1)

ax.set_title(fr'Calculated Eigenvalues for square membrane, $N={N}$')
ax.set_xlabel(r'Eigenvalue number $n$')
ax.set_ylabel(r'Eigenvalue $k^2$')
ax.grid()

axin = fig.add_axes([0.18, 0.53, 0.38, 0.30])
axin.set_title('First Few')
axin.grid()

ax.plot(N**2 * eigvals, label=fr'Calculated')
ax.plot(eigs, label=fr'Analytical')

axin.plot(N**2 * eigvals[:10])
axin.plot(eigs[:10])

ax.legend(loc='lower right')
plt.savefig('images/spec.png')
plt.show()
"""

"""
#------------------------------------------------------------------------------
# Testing speed of implementations

# different N
N_num = 30
N_lst = range(2, N_num)

# Set up parameters
tol = 10**(-5)

# Initizalize list of times
times_N = {'Jac': [], 'Eigh': [], 'Pwr': [], 'Inv': [], }

for N in N_lst:

	print(f'Calculating for N={N}')

	A = make_A(N)
		
	start = timer()
	eigvals, eigvecs = pwr_iteration(A, N)
	end = timer()
	times_N['Pwr'].append(end - start)

	Ainv = la.inv(A)
	start = timer()
	eigvals, eigvecs = pwr_iteration(Ainv, N)
	end = timer()
	times_N['Inv'].append(end - start)

	#start = timer()
	#eigvals, eigvecs = jacobi(A, N)
	#end = timer()
	#times_N['Jac'].append(end - start)

	start = timer()
	eigvals, eigvecs = la.eigh(A)
	end = timer()
	times_N['Eigh'].append(end - start)
	
# Plotting
fig, ax = plt.subplots(1)

plt.xlabel(r'Number of Points $N$')
plt.ylabel(r'Evaluation Time $t$ [s]')
plt.title(fr'Evaluation Time for Different N, homogeneous')
plt.xscale('log')
plt.yscale('log')
plt.grid()

#plt.plot(N_lst, times_N['Jac'], 'o-', markersize=0.8, label=r'Jacobi')
plt.plot(N_lst, times_N['Pwr'], 'o-', markersize=0.8, label=r'Power Iteration')
plt.plot(N_lst, times_N['Inv'], 'o-', markersize=0.8, label=r'Inverse Iteration')
plt.plot(N_lst, times_N['Eigh'], 'o-', markersize=0.8, label=r'\texttt{scipy.linalg.eigh}')

print(times_N)

plt.legend(loc='best')
plt.savefig('images/times_alt.png')
plt.show()
"""

"""
#------------------------------------------------------------------------------
# Testing convergence of power iteration

N = 50
n = 7
tol = 10**(-5)
A = make_A(N)

alleigs, eigvecs = pwr_iteration(A, n, saveall=True)

# Plotting
fig, ax = plt.subplots(1)

plt.xlabel(r'Iteration number $m$')
plt.ylabel(r'Eigenvalue Deviation $|k^2 - k_{\mathrm{final}}^2|$')
plt.title(fr'Convergence of Eigenvalues, Power Iteration, $N={N}$')
plt.yscale('log')
plt.grid()

plt.axhline(tol, color='k', label='Tolerance')
for i in range(n):
	plt.plot(np.abs((alleigs[i][:-1] - alleigs[i][-1])), 'o-', markersize=0.8, label=fr'Eigenvalue ${i+1}$')

plt.legend(loc='best')
plt.savefig('images/pwr_conv_alt.png')
plt.show()

#------------------------------------------------------------------------------
# Testing convergence of inverse iteration

N = 50
n = 7
tol = 10**(-5)
A = make_A(N)
Ainv = la.inv(A)

alleigs, eigvecs = pwr_iteration(Ainv, n, saveall=True)

# Plotting
fig, ax = plt.subplots(1)

plt.xlabel(r'Iteration number $m$')
plt.ylabel(r'Eigenvalue Deviation $|k^2 - k_{\mathrm{final}}^2|$')
plt.title(fr'Convergence of Eigenvalues, Inverse Iteration, $N={N}$')
plt.yscale('log')
plt.grid()

plt.axhline(tol, color='k', label='Tolerance')
for i in range(n):
	plt.plot(np.abs((alleigs[i][:-1] - alleigs[i][-1])), 'o-', markersize=0.8, label=fr'Eigenvalue ${i+1}$')

plt.legend(loc='best')
plt.savefig('images/inv_conv_alt.png')
plt.show()

#------------------------------------------------------------------------------
# Testing convergence of Jacobi
N = 15
tol = 10**(-5)
A = make_A(N)

_, _, offall = jacobi(A, conv=True)

# Plotting
fig, ax = plt.subplots(1)

plt.xlabel(r'Iteration number $m$')
plt.ylabel(r'Offset $\mathrm{off}(A)$')
plt.title(fr'Convergence of Jacobi, $N={N}$')
plt.yscale('log')
plt.grid()

plt.axhline(tol, color='k', label='Tolerance')
plt.plot(offall, 'o-', markersize=0.8)

plt.legend(loc='best')
plt.savefig('images/jac_conv.png')
plt.show()
"""

"""
#------------------------------------------------------------------------------
# Testing accuracy of implementations

N = 70
n = 20
tol = 10**(-5)
A = make_A(N)
Ainv = la.inv(A)

actualeigs = []
for i in range(1, N+1):
	for j in range(1, N+1):
		actualeigs.append(np.pi**2 * (i**2 + j**2))
actualeigs = np.sort(actualeigs)[:n]

eigvals_eigh, eigvecs_eigh = la.eigh(A, subset_by_index=(0, n-1))
eigvals_inv, eigvecs_inv = pwr_iteration(Ainv, n, tol=tol)

#sorted_ind = np.argsort(1 / eigvals_inv)
eigvals_inv = np.sort(1 / eigvals_inv)[:n]
#eigvecs_inv = eigvecs_inv[sorted_ind][:n]

# Plotting
fig, ax = plt.subplots(1)

ax.set_xlabel(r'Eigenvalue number $n$')
ax.set_ylabel(r'Eigenvalue Deviation $|k_{\mathrm{calc}}^2 / k_{\mathrm{real}}^2 - 1|$')
ax.set_title(fr'Accuracy of Methods, $\epsilon={tol}$, $N={N}$')
ax.set_yscale('log')
ax.set_xlim(-0.9, 22)
ax.set_ylim(0.0272, 0.03196)
ax.grid()

axin = fig.add_axes([0.51, 0.17, 0.38, 0.30])
axin.set_title('Method Difference')
axin.grid()

ax.plot(np.abs(N**2 * eigvals_inv / actualeigs - 1) , 'o-', markersize=3, label=r'Inverse Iteration')
ax.plot(np.abs(N**2 * eigvals_eigh / actualeigs - 1) , 'o-', markersize=3, label=r'\texttt{scipy.linalg.eigh}')

axin.plot(np.abs(N**2 * (eigvals_eigh - eigvals_inv)), 'o-', color='k', markersize=3)

ax.legend(loc='upper left')
plt.savefig('images/accuracy_fast.png')
plt.show()


fig, ax = plt.subplots(1)

ax.set_title(fr'Eigenfunction 1 Accuracy $|u - u_{{\mathrm{{real}}}}|$, N={N}')
ax.set_xlabel(r'$x$ coordinate')
ax.set_ylabel(r'$y$ coordinate')

im = ax.matshow(np.abs(vec_to_mat(eigvecs_eigh[:,0]) - anal(1, 1, N)))

clb = plt.colorbar(im)
clb.ax.set_title(r'Amp. $A$')
plt.savefig('eig_1_error.png')
plt.show()


fig, ax = plt.subplots(1)

ax.set_title(fr'Eigenfunction 11 Accuracy $|u - u_{{\mathrm{{real}}}}|$, N={N}')
ax.set_xlabel(r'$x$ coordinate')
ax.set_ylabel(r'$y$ coordinate')

im = ax.matshow(np.abs(-vec_to_mat(eigvecs_eigh[:,11]) - anal(3, 3, N)))

clb = plt.colorbar(im)
clb.ax.set_title(r'Amp. $A$')
plt.savefig('eig_11_error.png')
plt.show()
"""



