from matplotlib.pyplot import *
from scipy import integrate
from numpy import *

# a simple Runge-Kutta integrator for multiple dependent variables and one independent variable

def rungekutta4(yprime, time, y0):
    # yprime is a list of functions, y0 is a list of initial values of y
    # time is a list of t-values at which solutions are computed
    #
    # Dependency: numpy

    N = len(time)

    y = array([thing*ones(N) for thing in y0]).T

    for ii in range(N-1):
        dt = time[ii+1] - time[ii]
        k1 = dt*yprime(y[ii], time[ii])
        k2 = dt*yprime(y[ii] + 0.5*k1, time[ii] + 0.5*dt)
        k3 = dt*yprime(y[ii] + 0.5*k2, time[ii] + 0.5*dt)
        k4 = dt*yprime(y[ii] + k3, time[ii+1])
        y[ii+1] = y[ii] + (k1 + 2.0*(k2 + k3) + k4)/6.0

    return y

# Miscellaneous functions
n= 1.0/3.0
kappa1 = 0.1
kappa2 = 0.1
kappa3 = 0.1
def total_energy(valpair):
    (x, y, px, py) = tuple(valpair)
    return .5*(px**2 + py**2) + (1.0/(1.0*(n+1)))*(kappa1*np.absolute(x)**(n+1)+kappa2*np.absolute(y-x)**(n+1)+kappa3*np.absolute(y)**(n+1))

def pqdot(valpair, tval):
    # input: [x, y, px, py], t
    # takes a pair of x and y values and returns \dot{p} according to the Hamiltonian
    (x, y, px, py) = tuple(valpair)
    return np.array([px, py, -kappa1*np.sign(x)*np.absolute(x)**n+kappa2*np.sign(y-x)*np.absolute(y-x)**n, kappa2*np.sign(y-x)*np.absolute(y-x)**n-kappa3*np.sign(y)*np.absolute(y)**n]).T

def findcrossings(data, data1):
    # returns indices in 1D data set where the data crossed zero. Useful for generating Poincare map at 0
    prb = list()
    for ii in range(len(data)-1):
        if (((data[ii] > 0) and (data[ii+1] < 0)) or ((data[ii] < 0) and (data[ii+1] > 0))) and data1[ii] > 0:
            prb.append(ii)
    return array(prb)

def refine_crossing(a,b):
    tf = -a[0]/a[2]
    while abs(b[0])>1e-6:
        b = odeint(pqdot, a, [0,tf], atol=1e-8, rtol=1e-6)[-1];
        # Newton step using that b[0]=x(tf) and b[2]=x'(tf)
        tf -= b[0]/b[2]
    return [ b[1], b[3] ]

t = linspace(0, 1000.0, 2000+1)
print ("step size is " + str(t[1]-t[0]))

# Representative initial conditions for E=1
E = 2
x0=0
y0=0
E_kin = E-total_energy([x0,y0,0,0])
init_cons = [[x0, y0, (2*E_kin-py**2)**0.5, py] for py in np.linspace(-10,10,8)]

outs = [ integrate.odeint(pqdot, con, t, atol=1e-9, rtol=1e-8) for con in init_cons[:8] ]


# plot the results
fig1 = figure(1)
for ii in range(4):
    subplot(2, 2, ii+1)
    plot(outs[ii][:,1],outs[ii][:,3])
    ylabel("py")
    xlabel("y")
    title("Full trajectory projected onto the plane")

fig1.suptitle('Full trajectories E = 1', fontsize=10)


# Plot Poincare sections at x=0 and px>0
fig2 = figure(2)
for ii in range(8):
    #subplot(4, 2, ii+1)
    xcrossings = findcrossings(outs[ii][:,0], outs[ii][:,3])
    ycrossings = [ refine_crossing(outs[ii][cross], outs[ii][cross+1]) for cross in xcrossings]
    yints, pyints = array(ycrossings).T
    plot(yints, pyints,'.')
    ylabel("py")
    xlabel("y")
    title("Poincare section x = 0")
