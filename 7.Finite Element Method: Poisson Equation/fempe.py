import numpy as np
import scipy.linalg as la
import scipy.spatial as sp

import matplotlib.pyplot as plt
from matplotlib import rc
from FitterPlotter import FitterPlotter

from timeit import default_timer as timer


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# Define the interior of the region
def mask(x, y, shape):

	if shape == 'square':
		cond1 = x >= -1
		cond2 = x <= 1
		cond3 = y >= -1
		cond4 = y <= 1

		return cond1 and cond2 and cond3 and cond4
	
	elif shape == 'circle':
		cond1 = (x**2 + y**2 <= 1)

		return cond1

	elif shape == 'circlebar':
		cond1 = ((x - 1/2)**2 + (y - 1/2)**2 <= 1/4)
		cond2 = x - 1/2 > 0 and np.abs(y - 1/2) < 0.05

		return cond1 and (not cond2)

	elif shape == 'semicircle':
		cond1 = (x**2 + (y + 1/2)**2 <= 1)
		cond2 = y >= -1/2

		return cond1 and cond2
	
	elif shape == 'birdhouse':
		cond1 = (x < 1/3-1/2 and y < 1/3-1/2)
		cond2 = (x > 2/3-1/2 and y < 1/3-1/2)
		cond3 = (y > x + 2/3)
		cond4 = (y > -x + 4/3-1 and y > 2/3-1/2)

		return not (cond1 or cond2 or cond3 or cond4)

	elif shape == 'astroid':
		cond1 = (np.abs(x - 1/2)**(2/3) + np.abs(y - 1/2)**(2/3) <= (1/2)**(2/3))

		return cond1

	else:
		print(f'The given shape "{shape}" is not implemented.')
		raise Exception

# Create a mesh and triangulation
def trimesh(shape, n, m, mtype='basic', rho=None):

	if mtype == 'advanced' and rho == None:
		rho = lambda x: 1 - (1 - x)**1.5
	
	# generate a layer 
	def layer(n, shape, s0=0):
		lyr = []
		if shape == 'circle':
			for s in np.linspace(0, 2 * np.pi, n, endpoint=False):
				p = [np.cos(s + s0), np.sin(s + s0)]
				lyr.append(p)

		elif shape == 'semicircle':
			for s in np.linspace(0, np.pi + 2, n, endpoint=False):
				if s < np.pi:
					p = [np.cos(s), np.sin(s) - 1/2]
				elif np.pi <= s:
					p = [s - 1 - np.pi, -1/2]
				lyr.append(p)

		elif shape == 'square':
			for s in np.linspace(0, 8, n, endpoint=False):
				if 0 <= s < 2:
					p = [-1 + s, -1]
				elif 2 <= s <= 4:
					p = [1, s - 3]
				elif 4 <= s < 6:
					p = [5 - s, 1]
				elif 6 <= s < 8:
					p = [-1, 7 - s]
				lyr.append(p)
		elif shape == 'birdhouse':
			for s in np.linspace(0, (8 + 2**(3/2))/3, n, endpoint=False):
				if 0 <= s < 1/3:
					p = [0-1/2, 2/3 - s-1/2]
				elif 1/3 <= s < 2/3:
					p = [s - 1/3-1/2, 1/3-1/2]
				elif 2/3 <= s < 1:
					p = [1/3-1/2, 1 - s-1/2]
				elif 1 <= s < 4/3:
					p = [s - 2/3-1/2, 0-1/2]
				elif 4/3 <= s < 5/3:
					p = [2/3-1/2, s - 4/3-1/2]
				elif 5/3 <= s < 2:
					p = [s - 1-1/2, 1/3-1/2]
				elif 2 <= s < 7/3:
					p = [1-1/2, s - 5/3-1/2]
				elif 7/3 <= s < 8/3:
					p = [10/3 - s-1/2, 2/3-1/2]
				elif 8/3 <= s < (8 + 2**(1/2))/3:
					p = [-(2)**(-1/2) * (s - 8/3) + 2/3-1/2, (2)**(-1/2) * (s - 8/3) + 2/3-1/2]
				elif s >= (8 + 2**(1/2))/3:
					p = [-(2)**(-1/2) * (s - (8 + 2**(1/2))/3) + 1/3-1/2, -(2)**(-1/2) * (s - (8 + 2**(1/2))/3) + 1-1/2]
				lyr.append(p)


		return np.array(lyr)

	# add the boundary
	bound = layer(4*n, shape)

	if mtype == 'advanced':

		# initialize domain
		domain = np.array([[0, 0]])
		
		# construct the domain
		for i, f in enumerate(rho(np.linspace(1, 0, m, endpoint=False))):
			domain = np.concatenate((domain, f * layer(int(f * n), shape,
				s0=np.pi*i/(n))))

	elif mtype == 'basic':

		# construct the domain
		domain = list(bound)
		xmin = np.min(bound[:,0])
		xmax = np.max(bound[:,0])
		ymin = np.min(bound[:,1])
		ymax = np.max(bound[:,1])
		for x in np.linspace(xmin, xmax, n):
			for y in np.linspace(ymin, ymax, m):
				if mask(x, y, shape) and [x, y] not in bound:
					domain.append([x, y])
		domain = np.array(domain)

	# create triangulation
	tri = sp.Delaunay(domain)
	
	return domain, bound, tri
	
# Solve for the flow
def solve_flow(cN, delD, cT):

	# calculate area of triangle
	def tarea(t):
		# global indices of trinagle
		glb_inds = cT[t,:]

		# x and y coordinates of vertices
		x = cN[glb_inds,0]
		y = cN[glb_inds,1]

		# area of triangle by formula
		return np.abs((x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) +
			x[2] * (y[0] - y[1])) / 2)
		
	# calculate matrix matrix A_{mn}^{t}
	def make_A(t, m, n):
		# global indices of trinagle
		glb_inds = cT[t,:]

		# x and y coordinates of vertices
		x = cN[glb_inds,0]
		y = cN[glb_inds,1]

		# Dirichlet condition for boundary
		if np.any(np.all(delD==[x[m], y[m]], axis=1)) or np.any(np.all(delD==[x[n], y[n]], axis=1)): return 0

		# calcualte area of triangle
		area = tarea(t)

		# return term by formula
		return (((y[(m+1)%3] - y[(m+2)%3]) * (y[(n+1)%3] - y[(n+2)%3]) +
			(x[(m+2)%3] - x[(m+1)%3]) * (x[(n+2)%3] - x[(n+1)%3])) / (4*area))

	# calculate vector g^{t}
	def make_g(t, m):
		# global indices of trinagle
		glb_inds = cT[t,:]

		# x and y coordinates of vertices
		x = cN[glb_inds,0]
		y = cN[glb_inds,1]

		# Dirichlet condition for boundary
		if np.any(np.all(delD==[x[m], y[m]], axis=1)): return 0

		# return terms from formula
		return (((x[(m+1)%3] - x[m%3]) * (y[(m+2)%3] - y[m%3]) -
			(x[(m+2)%3] - x[m%3]) * (y[(m+1)%3] - y[m%3])) / 6)

	# initialize matrix and vectors
	S = np.zeros((len(cN), len(cN)))
	c = np.zeros((len(cN)))
	g = np.zeros((len(cN)))

	# build matrix S and vector g
	for t in range(len(cT)):

		# ignore degenerate triangles
		if tarea(t) == 0: continue

		# loop over local indices
		for m in range(3):
			
			# matrix S
			for n in range(3):
				S[cT[t,m],cT[t,n]] += make_A(t, m, n)
			
			# vector g
			g[cT[t,m]] += make_g(t, m)

	# remove Dirichlet zeros
	inds = np.any(S, axis=0)
	S = S[inds,:][:,inds]
	g = g[inds]
	
	# solve system of linear equations and add Dirichlet zeros
	cp = np.linalg.solve(S, g)
	c[inds] = cp

	return c

def calc_C(cN, cT, c, shape):
	Phi = 0
	Area = 0

	for t in range(len(cT)):
		# global indices of trinagle
		glb_inds = cT[t,:]

		# x and y coordinates of vertices
		x = cN[glb_inds,0]
		y = cN[glb_inds,1]

		# area of triangle by formula
		tarea = np.abs((x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) +
			x[2] * (y[0] - y[1])) / 2)

		Area += tarea

		# average value of velocity over triangle
		u_avg = np.sum(c[glb_inds]) / 3

		# increment C
		Phi += tarea * u_avg

	print(8 * np.pi * Phi / (5/9)**2)

	return 8 * np.pi * Phi / Area**2


def draw_profile(cN, tri, shape, plot_type, savefig=False):
	fig, ax = plt.subplots(1)
	plt.title(fr'Flow Profile, $N={len(cN)}$')
	plt.xlabel(r'$x$ coordinate')
	plt.ylabel(r'$y$ coordinate')
	ax.set_aspect('equal')


	if plot_type == 'scatter':
		sct = plt.scatter(cN[:,0], cN[:,1], c=c)
		clb = plt.colorbar(sct, shrink=0.7, aspect=9)

	if plot_type == 'contourf':
		cnt = plt.tricontourf(cN[:,0], cN[:,1], tri.simplices, c, levels=10)
		clb = plt.colorbar(cnt, shrink=0.7, aspect=9)

	clb.ax.set_title(r'Velocity $u$')

	if savefig: plt.savefig(f'images/profile_{shape}_{plot_type}_{len(cN)}.png')

# advanced 500, 70
n = 24
m = 20
shape = 'square'

lyr = []
bound = []

hexx = [0.0005036630037, 0.0005036630037, 0.0005036630037, 0.0005036630037, 0.0005036630037, 0.0005036630037, 0.0005036630037, 0.0005036630037, 0.0005036630037, 0.0005036630037, 0.0005036630037, 0.02500915751, 0.02500915751, 0.02500915751, 0.02500915751, 0.02500915751, 0.02500915751, 0.02500915751, 0.02500915751, 0.02500915751, 0.02500915751, 0.04951465201, 0.04951465201, 0.04951465201, 0.04951465201, 0.04951465201, 0.04951465201, 0.04951465201, 0.04951465201, 0.04951465201, 0.04951465201, 0.04951465201, 0.07402014652, 0.07402014652, 0.07402014652, 0.07402014652, 0.07402014652, 0.07402014652, 0.07402014652, 0.07402014652, 0.07402014652, 0.07402014652, 0.1005677656, 0.1005677656, 0.1005677656, 0.1005677656, 0.1005677656, 0.1005677656, 0.1005677656, 0.1005677656, 0.1005677656, 0.1005677656, 0.1005677656, 0.1250732601, 0.1250732601, 0.1250732601, 0.1250732601, 0.1250732601, 0.1250732601, 0.1250732601, 0.1250732601, 0.1250732601, 0.1250732601, 0.1495787546, 0.1495787546, 0.1495787546, 0.1495787546, 0.1495787546, 0.1495787546, 0.1495787546, 0.1495787546, 0.1495787546, 0.1495787546, 0.1495787546, 0.1740842491, 0.1740842491, 0.1740842491, 0.1740842491, 0.1740842491, 0.1740842491, 0.1740842491, 0.1740842491, 0.1740842491, 0.1740842491, 0.2006318681, 0.2006318681, 0.2006318681, 0.2006318681, 0.2006318681, 0.2006318681, 0.2006318681, 0.2006318681, 0.2006318681, 0.2006318681, 0.2006318681, 0.2251373626, 0.2251373626, 0.2251373626, 0.2251373626, 0.2251373626, 0.2251373626, 0.2251373626, 0.2251373626, 0.2251373626, 0.2251373626, 0.2496428571, 0.2496428571, 0.2496428571, 0.2496428571, 0.2496428571, 0.2496428571, 0.2496428571, 0.2496428571, 0.2496428571, 0.2496428571, 0.2496428571, 0.2761904762, 0.2761904762, 0.2761904762, 0.2761904762, 0.2761904762, 0.2761904762, 0.2761904762, 0.2761904762, 0.2761904762, 0.2761904762, 0.3006959707, 0.3006959707, 0.3006959707, 0.3006959707, 0.3006959707, 0.3006959707, 0.3006959707, 0.3006959707, 0.3006959707, 0.3006959707, 0.3006959707, 0.3252014652, 0.3252014652, 0.3252014652, 0.3252014652, 0.3252014652, 0.3252014652, 0.3252014652, 0.3252014652, 0.3252014652, 0.3252014652, 0.3497069597, 0.3497069597, 0.3497069597, 0.3497069597, 0.3497069597, 0.3497069597, 0.3497069597, 0.3497069597, 0.3497069597, 0.3497069597, 0.3497069597, 0.3762545788, 0.3762545788, 0.3762545788, 0.3762545788, 0.3762545788, 0.3762545788, 0.3762545788, 0.3762545788, 0.3762545788, 0.3762545788, 0.4007600733, 0.4007600733, 0.4007600733, 0.4007600733, 0.4007600733, 0.4007600733, 0.4007600733, 0.4007600733, 0.4007600733, 0.4007600733, 0.4007600733, 0.4252655678, 0.4252655678, 0.4252655678, 0.4252655678, 0.4252655678, 0.4252655678, 0.4252655678, 0.4252655678, 0.4252655678, 0.4252655678, 0.4497710623, 0.4497710623, 0.4497710623, 0.4497710623, 0.4497710623, 0.4497710623, 0.4497710623, 0.4497710623, 0.4497710623, 0.4497710623, 0.4497710623, 0.4763186813, 0.4763186813, 0.4763186813, 0.4763186813, 0.4763186813, 0.4763186813, 0.4763186813, 0.4763186813, 0.4763186813, 0.4763186813, 0.5008241758, 0.5008241758, 0.5008241758, 0.5008241758, 0.5008241758, 0.5008241758, 0.5008241758, 0.5008241758, 0.5008241758, 0.5008241758, 0.5008241758, 0.5253296703, 0.5253296703, 0.5253296703, 0.5253296703, 0.5253296703, 0.5253296703, 0.5253296703, 0.5253296703, 0.5253296703, 0.5253296703, 0.5518772894, 0.5518772894, 0.5518772894, 0.5518772894, 0.5518772894, 0.5518772894, 0.5518772894, 0.5518772894, 0.5518772894, 0.5518772894, 0.5518772894, 0.5763827839, 0.5763827839, 0.5763827839, 0.5763827839, 0.5763827839, 0.5763827839, 0.5763827839, 0.5763827839, 0.5763827839, 0.5763827839, 0.6008882784, 0.6008882784, 0.6008882784, 0.6008882784, 0.6008882784, 0.6008882784, 0.6008882784, 0.6008882784, 0.6008882784, 0.6008882784, 0.6008882784, 0.6253937729, 0.6253937729, 0.6253937729, 0.6253937729, 0.6253937729, 0.6253937729, 0.6253937729, 0.6253937729, 0.6253937729, 0.6253937729, 0.6519413919, 0.6519413919, 0.6519413919, 0.6519413919, 0.6519413919, 0.6519413919, 0.6519413919, 0.6519413919, 0.6519413919, 0.6519413919, 0.6519413919, 0.6764468864, 0.6764468864, 0.6764468864, 0.6764468864, 0.6764468864, 0.6764468864, 0.6764468864, 0.6764468864, 0.6764468864, 0.6764468864, 0.700952381, 0.700952381, 0.700952381, 0.700952381, 0.700952381, 0.700952381, 0.700952381, 0.700952381, 0.700952381, 0.700952381, 0.700952381, 0.7254578755, 0.7254578755, 0.7254578755, 0.7254578755, 0.7254578755, 0.7254578755, 0.7254578755, 0.7254578755, 0.7254578755, 0.7254578755, 0.7520054945, 0.7520054945, 0.7520054945, 0.7520054945, 0.7520054945, 0.7520054945, 0.7520054945, 0.7520054945, 0.7520054945, 0.7520054945, 0.7520054945, 0.776510989, 0.776510989, 0.776510989, 0.776510989, 0.776510989, 0.776510989, 0.776510989, 0.776510989, 0.776510989, 0.776510989, 0.8010164835, 0.8010164835, 0.8010164835, 0.8010164835, 0.8010164835, 0.8010164835, 0.8010164835, 0.8010164835, 0.8010164835, 0.8010164835, 0.8010164835, 0.8275641026, 0.8275641026, 0.8275641026, 0.8275641026, 0.8275641026, 0.8275641026, 0.8275641026, 0.8275641026, 0.8275641026, 0.8275641026, 0.8520695971, 0.8520695971, 0.8520695971, 0.8520695971, 0.8520695971, 0.8520695971, 0.8520695971, 0.8520695971, 0.8520695971, 0.8520695971, 0.8520695971, 0.8765750916, 0.8765750916, 0.8765750916, 0.8765750916, 0.8765750916, 0.8765750916, 0.8765750916, 0.8765750916, 0.8765750916, 0.8765750916, 0.9010805861, 0.9010805861, 0.9010805861, 0.9010805861, 0.9010805861, 0.9010805861, 0.9010805861, 0.9010805861, 0.9010805861, 0.9010805861, 0.9010805861, 0.9276282051, 0.9276282051, 0.9276282051, 0.9276282051, 0.9276282051, 0.9276282051, 0.9276282051, 0.9276282051, 0.9276282051, 0.9276282051, 0.9521336996, 0.9521336996, 0.9521336996, 0.9521336996, 0.9521336996, 0.9521336996, 0.9521336996, 0.9521336996, 0.9521336996, 0.9521336996, 0.9521336996, 0.9766391941, 0.9766391941, 0.9766391941, 0.9766391941, 0.9766391941, 0.9766391941, 0.9766391941, 0.9766391941, 0.9766391941, 0.9766391941, 1.001144689, 1.001144689, 1.001144689, 1.001144689, 1.001144689, 1.001144689, 1.001144689, 1.001144689, 1.001144689, 1.001144689, 1.001144689]
hexy = [0.9966825888, 0.8962952353, 0.7982424715, 0.6978551181, 0.5998023543, 0.4994150008, 0.401362237, 0.3009748836, 0.2005875301, 0.1025347663, 0.002147412889, 0.9453831888, 0.847330425, 0.7469430716, 0.6488903078, 0.5485029543, 0.4504501905, 0.3500628371, 0.2520100733, 0.1516227198, 0.05356995601, 0.996805732, 0.8964183785, 0.7983656147, 0.6979782613, 0.5999254975, 0.499538144, 0.4014853802, 0.3010980268, 0.2007106733, 0.1026579095, 0.002270556077, 0.945506332, 0.8474535682, 0.7470662148, 0.649013451, 0.5486260975, 0.4505733337, 0.3501859803, 0.2521332164, 0.151745863, 0.0536930992, 0.9969340061, 0.8965466527, 0.7984938889, 0.6981065354, 0.6000537716, 0.4996664182, 0.4016136544, 0.3012263009, 0.2008389475, 0.1027861837, 0.002398830232, 0.9456346062, 0.8475818424, 0.7471944889, 0.6491417251, 0.5487543717, 0.4507016079, 0.3503142544, 0.2522614906, 0.1518741372, 0.05382137335, 0.9970571493, 0.8966697959, 0.798617032, 0.6982296786, 0.6001769148, 0.4997895614, 0.4017367975, 0.3013494441, 0.2009620907, 0.1029093269, 0.00252197342, 0.9457577494, 0.8477049856, 0.7473176321, 0.6492648683, 0.5488775149, 0.450824751, 0.3504373976, 0.2523846338, 0.1519972804, 0.05394451654, 0.9971854235, 0.89679807, 0.7987453062, 0.6983579528, 0.6003051889, 0.4999178355, 0.4018650717, 0.3014777183, 0.2010903648, 0.103037601, 0.002650247575, 0.9458860235, 0.8478332597, 0.7474459063, 0.6493931425, 0.549005789, 0.4509530252, 0.3505656718, 0.2525129079, 0.1521255545, 0.05407279069, 0.9973085666, 0.8969212132, 0.7988684494, 0.698481096, 0.6004283321, 0.5000409787, 0.4019882149, 0.3016008615, 0.201213508, 0.1031607442, 0.002773390763, 0.9460142977, 0.8479615339, 0.7475741804, 0.6495214166, 0.5491340632, 0.4510812994, 0.3506939459, 0.2526411821, 0.1522538287, 0.05420106485, 0.9974368408, 0.8970494874, 0.7989967235, 0.6986093701, 0.6005566063, 0.5001692529, 0.402116489, 0.3017291356, 0.2013417822, 0.1032890184, 0.002901664918, 0.9461374409, 0.848084677, 0.7476973236, 0.6496445598, 0.5492572064, 0.4512044425, 0.3508170891, 0.2527643253, 0.1523769719, 0.05432420804, 0.997559984, 0.8971726306, 0.7991198667, 0.6987325133, 0.6006797495, 0.500292396, 0.4022396322, 0.3018522788, 0.2014649254, 0.1034121615, 0.003024808106, 0.946265715, 0.8482129512, 0.7478255978, 0.6497728339, 0.5493854805, 0.4513327167, 0.3509453633, 0.2528925994, 0.152505246, 0.05445248219, 0.9976882581, 0.8973009047, 0.7992481409, 0.6988607875, 0.6008080236, 0.5004206702, 0.4023679064, 0.3019805529, 0.2015931995, 0.1035404357, 0.003153082261, 0.9463888582, 0.8483360944, 0.747948741, 0.6498959771, 0.5495086237, 0.4514558599, 0.3510685065, 0.2530157426, 0.1526283892, 0.05457562538, 0.9978114013, 0.8974240479, 0.7993712841, 0.6989839306, 0.6009311668, 0.5005438134, 0.4024910496, 0.3021036961, 0.2017163427, 0.1036635789, 0.00327622545, 0.9465171324, 0.8484643685, 0.7480770151, 0.6500242513, 0.5496368979, 0.451584134, 0.3511967806, 0.2531440168, 0.1527566634, 0.05470389953, 0.9979396755, 0.897552322, 0.7994995582, 0.6991122048, 0.601059441, 0.5006720875, 0.4026193237, 0.3022319703, 0.2018446169, 0.103791853, 0.003404499604, 0.9466402756, 0.8485875117, 0.7482001583, 0.6501473945, 0.549760041, 0.4517072772, 0.3513199238, 0.25326716, 0.1528798065, 0.05482704272, 0.9980679496, 0.8976805962, 0.7996278324, 0.699240479, 0.6011877151, 0.5008003617, 0.4027475979, 0.3023602444, 0.201972891, 0.1039201272, 0.003532773759, 0.9467685497, 0.8487157859, 0.7483284325, 0.6502756686, 0.5498883152, 0.4518355514, 0.3514481979, 0.2533954341, 0.1530080807, 0.05495531688, 0.9981910928, 0.8978037394, 0.7997509756, 0.6993636221, 0.6013108583, 0.5009235049, 0.4028707411, 0.3024833876, 0.2020960342, 0.1040432704, 0.003655916947, 0.9468916929, 0.8488389291, 0.7484515756, 0.6503988118, 0.5500114584, 0.4519586946, 0.3515713411, 0.2535185773, 0.1531312239, 0.05507846007, 0.998319367, 0.8979320135, 0.7998792497, 0.6994918963, 0.6014391325, 0.501051779, 0.4029990152, 0.3026116618, 0.2022243084, 0.1041715445, 0.003784191102, 0.9470199671, 0.8489672032, 0.7485798498, 0.650527086, 0.5501397325, 0.4520869687, 0.3516996153, 0.2536468515, 0.153259498, 0.05520673422, 0.9984425102, 0.8980551567, 0.8000023929, 0.6996150395, 0.6015622757, 0.5011749222, 0.4031221584, 0.302734805, 0.2023474515, 0.1042946877, 0.003907334291, 0.9471431102, 0.8490903464, 0.748702993, 0.6506502292, 0.5502628757, 0.4522101119, 0.3518227585, 0.2537699947, 0.1533826412, 0.05532987741, 0.9985707843, 0.8981834309, 0.8001306671, 0.6997433136, 0.6016905498, 0.5013031964, 0.4032504326, 0.3028630791, 0.2024757257, 0.1044229619, 0.004035608445, 0.9472713844, 0.8492186206, 0.7488312671, 0.6507785033, 0.5503911499, 0.4523383861, 0.3519510326, 0.2538982688, 0.1535109154, 0.05545815156, 0.9986939275, 0.8983065741, 0.8002538103, 0.6998664568, 0.601813693, 0.5014263396, 0.4033735758, 0.3029862223, 0.2025988689, 0.1045461051, 0.004158751634, 0.9473996585, 0.8493468947, 0.7489595413, 0.6509067775, 0.550519424, 0.4524666602, 0.3520793068, 0.254026543, 0.1536391895, 0.05558642572, 0.9988222017, 0.8984348482, 0.8003820844, 0.699994731, 0.6019419672, 0.5015546137, 0.4035018499, 0.3031144965, 0.202727143, 0.1046743792, 0.004287025788, 0.9475228017, 0.8494700379, 0.7490826845, 0.6510299207, 0.5506425672, 0.4525898034, 0.35220245, 0.2541496862, 0.1537623327, 0.05570956891, 0.9989453449, 0.8985579914, 0.8005052276, 0.7001178742, 0.6020651104, 0.5016777569, 0.4036249931, 0.3032376397, 0.2028502862, 0.1047975224, 0.004410168977, 0.9476510759, 0.8495983121, 0.7492109586, 0.6511581948, 0.5507708414, 0.4527180776, 0.3523307241, 0.2542779603, 0.1538906069, 0.05583784306, 0.999073619, 0.8986862656, 0.8006335018, 0.7002461483, 0.6021933845, 0.5018060311, 0.4037532673, 0.3033659138, 0.2029785604, 0.1049257966, 0.004538443131, 0.9477742191, 0.8497214553, 0.7493341018, 0.651281338, 0.5508939846, 0.4528412208, 0.3524538673, 0.2544011035, 0.1540137501, 0.05596098625, 0.9991967622, 0.8988094088, 0.8007566449, 0.7003692915, 0.6023165277, 0.5019291743, 0.4038764104, 0.303489057, 0.2031017036, 0.1050489398, 0.00466158632]
hexx = np.array(hexx)
hexy = np.array(hexy)

xmin = np.min(hexx)
xmax = np.max(hexx)
ymin = np.min(hexy)
ymax = np.max(hexy)
print(xmin, xmax, ymin, ymax)

hexx = (hexx - xmin) / xmax
hexy = (hexy - ymin) / ymax

for i in range(len(hexx)):
	lyr.append([hexx[i], hexy[i]])

	cond1 = np.abs(hexx[i] - xmin) < 10**(-2)
	cond2 = np.abs(hexx[i] - xmax) < 10**(-2)
	cond3 = np.abs(hexy[i] - ymin) < 10**(-2)
	cond4 = np.abs(hexy[i] - ymax) < 10**(-2)

	if cond1 or cond2 or cond3 or cond4:
		bound.append([hexx[i], hexy[i]])

points = np.array(lyr)
bound = np.array(bound)
print(len(points))
tri = sp.Delaunay(points)

fig, ax = plt.subplots(1)

ax.set_title(fr'Hexagonal Mesh for the Square')
ax.set_xlabel(r'$x$ coordinate')
ax.set_ylabel(r'$y$ coordinate')

sp.delaunay_plot_2d(tri, ax=ax)

#plt.savefig('images/hex_square_mesh.png')
plt.show()

# Solve for flow profile
c = solve_flow(points, bound, tri.simplices)

# Calculate Poiseuille coefficient
C = calc_C(points, tri.simplices, c, shape)

# Draw flow profile
draw_profile(points, tri, shape, 'contourf', savefig=True)
plt.show()
"""
# Create mesh and triangulation
points, bound, tri = trimesh(shape, n, m, 'basic')
print(len(points))

tri = sp.Delaunay(points)

fig, ax = plt.subplots(1)

ax.set_title(fr'Square Mesh for the Square')
ax.set_xlabel(r'$x$ coordinate')
ax.set_ylabel(r'$y$ coordinate')

sp.delaunay_plot_2d(tri, ax=ax)

plt.savefig('images/sq_square_mesh.png')
plt.show()

# Solve for flow profile
c = solve_flow(points, bound, tri.simplices)

# Calculate Poiseuille coefficient
C = calc_C(points, tri.simplices, c, shape)

# Draw flow profile
draw_profile(points, tri, shape, 'contourf')
plt.show()
"""

"""
# advanced 200, 20
n = 56
m = 10
shape = 'circle'

# Create mesh and triangulation
points, bound, tri = trimesh(shape, n, m, 'advanced')
print(len(points))

tri = sp.Delaunay(points)
fig, ax = plt.subplots(1)

ax.set_aspect('equal')
ax.set_title(fr'Procedure-generated Mesh for the Circle')
ax.set_xlabel(r'$x$ coordinate')
ax.set_ylabel(r'$y$ coordinate')

sp.delaunay_plot_2d(tri, ax=ax)

plt.savefig('images/proc_mesh.png')
plt.show()
# Solve for flow profile
c = solve_flow(points, bound, tri.simplices)

# Calculate Poiseuille coefficient
C = calc_C(points, tri.simplices, c, shape)

# Draw flow profile
draw_profile(points, tri, shape, 'contourf')
plt.show()


n = 20
m = 20
shape = 'circle'

# Create mesh and triangulation
points, bound, tri = trimesh(shape, n, m, 'basic')
print(len(points))


tri = sp.Delaunay(points)

fig, ax = plt.subplots(1)

ax.set_aspect('equal')
ax.set_title(fr'Square Mesh for the Circle')
ax.set_xlabel(r'$x$ coordinate')
ax.set_ylabel(r'$y$ coordinate')

sp.delaunay_plot_2d(tri, ax=ax)

plt.savefig('images/sq_mesh.png')
plt.show()
# Solve for flow profile
c = solve_flow(points, bound, tri.simplices)

# Calculate Poiseuille coefficient
C = calc_C(points, tri.simplices, c, shape)

# Draw flow profile
draw_profile(points, tri, shape, 'contourf')
plt.show()
"""

"""
#------------------------------------------------------------------------------
# Testing speed of implementation
shape = 'square'

# different n
num_lst = range(4, 90)

# Initizalize list
times = []
nodes = []
faces = []

for n in num_lst:

	print(f'Calculating for n={n}')

	points, bound, tri = trimesh(shape, n, n, 'basic')
	nodes.append(len(points))
	faces.append(len(tri.simplices))

	start = timer()
	solve_flow(points, bound, tri.simplices)
	end = timer()
	times.append(end - start)

print('nodes:', nodes)
print('faces:', faces)
print('times:', times)

xlabel = r'Number of Nodes $\ln(N_{N})$'
ylabel = r'Evaluation Time $\ln(t/[s])$'
title = fr'Evaluation Time for Different Numbers of Nodes'
savepath = 'images/times_nodes_fit.png'
textbox = {'params': [0], 'symbols': ['slope'], 'coords': [0.6, 0.3]}
FitterPlotter('lin', [1.0, 1.0], np.log(nodes), np.log(times), nstd=2,
	title=title, xlabel=xlabel, ylabel=ylabel,
	savepath=savepath, textbox=textbox)

xlabel = r'Number of Triangles $\ln(N_{T})$'
title = fr'Evaluation Time for Different Numbers of Triangles'
savepath = 'images/times_faces_fit.png'
FitterPlotter('lin', [1.0, 1.0], np.log(faces), np.log(times), nstd=2,
	title=title, xlabel=xlabel, ylabel=ylabel,
	savepath=savepath, textbox=textbox)

#------------------------------------------------------------------------------
"""

r"""
#------------------------------------------------------------------------------
# Plotting the flow profile and calculating C

# Parameters M and N
M = 100
N = 100

# calculate C and plot u 
_, _, C_lst = calc(M, N, to_calc_u=True, to_plot_u=True)

# print values and errors of C
for i in range(len(C_lst)):
	if i != len(C_lst) - 1:
		print(f'Guess {i+1}: {C_lst[i]} +- {np.abs(C_lst[i] - C_lst[i +1])}')
	else:
		print(f'Guess {i+1}: {C_lst[i]} +- ???')

#------------------------------------------------------------------------------
"""
"""
#------------------------------------------------------------------------------
# Testing accuracy of implementation
shape = 'circle'

# different n
num_lst = range(4, 90)

# Initizalize list
Cs = []
nodes = []

for n in num_lst:

	print(f'Calculating for n={n}')

	points, bound, tri = trimesh(shape, n, n, 'basic')
	nodes.append(len(points))

	# Solve for flow profile
	c = solve_flow(points, bound, tri.simplices)

	# Calculate Poiseuille coefficient
	C = calc_C(points, tri.simplices, c, shape)
	Cs.append(C)

print('nodes:', nodes)
print('Cs:', Cs)


# Plotting
fig, ax = plt.subplots(1)

ax.set_title(fr'Coefficient for Different Numbers of Nodes, circle, uniform')
ax.set_xlabel(r'Number of Nodes $N_N$')
ax.set_ylabel(r'Coefficient $C$')
ax.grid()

axin = fig.add_axes([0.35, 0.23, 0.5, 0.45])
axin.set_title(r'$|C_{\mathrm{real}} - C_{\mathrm{calc}}|$')
axin.grid()
axin.set_yscale('log')

ax.plot(nodes, Cs, 'o-', label='Calculated')
ax.axhline(1, color='k', label='Analytical')

Cs = np.array(Cs)
axin.plot(nodes, np.abs(Cs - 1))

ax.legend(loc='lower left')
plt.savefig('images/Cs_circle_uniform_alt.png')
plt.show()
"""

"""
# advanced 59, 10
n = 75
m = 75
shape = 'semicircle'

# Create mesh and triangulation
points, bound, tri = trimesh(shape, n, m, 'basic')
print(len(points))

tri = sp.Delaunay(points)
fig, ax = plt.subplots(1)

ax.set_aspect('equal')
ax.set_title(fr'Procedure-generated Mesh for the Semicircle')
ax.set_xlabel(r'$x$ coordinate')
ax.set_ylabel(r'$y$ coordinate')

sp.delaunay_plot_2d(tri, ax=ax)

plt.savefig('images/sq_mesh_semi.png')
plt.show()
# Solve for flow profile
c = solve_flow(points, bound, tri.simplices)

# Calculate Poiseuille coefficient
C = calc_C(points, tri.simplices, c, shape)

# Draw flow profile
draw_profile(points, tri, shape, 'contourf', savefig=True)
plt.show()
"""
"""
# advanced 59, 10
n = 300
m = 80
shape = 'birdhouse'

# Create mesh and triangulation
points, bound, tri = trimesh(shape, n, m, 'advanced')
print(len(points))

tri = sp.Delaunay(points)
fig, ax = plt.subplots(1)

ax.set_aspect('equal')
ax.set_title(fr'Square Mesh for the Birdhouse')
ax.set_xlabel(r'$x$ coordinate')
ax.set_ylabel(r'$y$ coordinate')

sp.delaunay_plot_2d(tri, ax=ax)

#plt.savefig('images/proc_mesh_bird.png')
plt.show()

# Solve for flow profile
c = solve_flow(points, bound, tri.simplices)

# Calculate Poiseuille coefficient
C = calc_C(points, tri.simplices, c, shape)

# Draw flow profile
draw_profile(points, tri, shape, 'contourf', savefig=True)
plt.show()
"""