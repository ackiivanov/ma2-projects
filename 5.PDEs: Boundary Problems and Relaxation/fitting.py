import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import odr


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
rc('text', usetex=True)

#Model function and object
def fit_function(beta, x):
	A, B = beta
	return A * x + B

model = odr.Model(fit_function)

#Data and data object
w = np.array([0.8275625283588963, 1.2165534798712103, 1.4843383070656362, 1.5551328356806084, 1.6087827103854795, 1.6684484310595435, 1.713337888359156, 1.7482510850226676, 1.7592175396035283, 1.763938958791798, 1.7922753840328536, 1.8145424098679928, 1.820789825519454, 1.8115091076514478, 1.8231985764875462, 1.834114675420635, 1.8433173540544594, 1.851764009781818, 1.8595696920265226, 1.866827233319856, 1.8733018031301882, 1.8796769203234778, 1.8851913932652993, 1.8905831346167017, 1.8951944186701835, 1.8995589602623755, 1.9037034690349952, 1.907458995956483, 1.9082876254165595, 1.9122086557902023, 1.915674398397806, 1.9190086872068828, 1.9217207196496475])
N = np.arange(2, 100, 3)

x = np.sin(np.pi/N)
y = 2 / w - 1

#Exclude first 4 points
data = odr.RealData(x, y)

#Set up ODR with model and data
odrm = odr.ODR(data, model, beta0=[1.0, 1.0])

#Run regression
out = odrm.run()

#Extract parameters
betaout = out.beta
betaerr = out.sd_beta
print('Fit parameters and their std deviations')
print('-------------------')
for i in range(len(betaout)):
	print('Parameter {}: '.format(i + 1) + str(betaout[i]) + ' +- ' + str(betaerr[i]))

E1 = betaout[0]
F1 = betaout[1]

#Fit curve and confidence intervals
nstd = 3.0
betaout_up = betaout + nstd * betaerr
betaout_dw = betaout - nstd * betaerr

x_dummy = np.linspace(min(x), max(x), 1000)
fit = fit_function(betaout, x_dummy)
fit_up = fit_function(betaout_up, x_dummy)
fit_dw = fit_function(betaout_dw, x_dummy)

#Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Number of Points $\sin(\pi/N)$')
plt.ylabel(r'Optimal Parameter $2/\omega_0 - 1$')
plt.title(fr'Optimal Parameter for birdhouse')
plt.xscale('log')
plt.yscale('log')
plt.ylim(10**(-2), 2)
plt.grid()

plt.plot(x, y, 'o', color='#1f77b4', label='SOR')
plt.plot(x_dummy, fit, color='#1f77b4')
axes.fill_between(x_dummy, fit_up, fit_dw, color='#1f77b4', alpha=0.25, label='Confidence interval')

model = odr.Model(fit_function)

#Data and data object
w = np.array([0.8275625283588963, 1.2778543788721357, 1.4856538688152785, 1.5310235380565653, 1.6225594518849324, 1.6830382344345751, 1.7252965682349024, 1.7607122647724784, 1.7894021461017193, 1.8124681160413985, 1.8320981089090211, 1.848792325744076, 1.8634356117674558, 1.8754411301516583, 1.8858150372381144, 1.8951050124200461, 1.902992129304228, 1.9098126612804298, 1.9154897198999137, 1.920618637568026, 1.925275145754652, 1.9295216393227206, 1.9334099885133853, 1.9364811683670173, 1.9397768906708563, 1.9423233706393421, 1.9448417556608049, 1.9474736587052, 1.9494259120423694, 1.951410256663341, 1.9209366453963757, 1.9232659344573246, 1.9256674048985747])
N = np.arange(2, 100, 3)

x = np.sin(np.pi/N)
y = 2 / w - 1

#Exclude first 4 points
data = odr.RealData(x, y)

#Set up ODR with model and data
odrm = odr.ODR(data, model, beta0=[1.0, 1.0])

#Run regression
out = odrm.run()

#Extract parameters
betaout = out.beta
betaerr = out.sd_beta
print('Fit parameters and their std deviations')
print('-------------------')
for i in range(len(betaout)):
	print('Parameter {}: '.format(i + 1) + str(betaout[i]) + ' +- ' + str(betaerr[i]))

E2 = betaout[0]
F2 = betaout[1]

#Fit curve and confidence intervals
nstd = 3.0
betaout_up = betaout + nstd * betaerr
betaout_dw = betaout - nstd * betaerr

x_dummy = np.linspace(min(x), max(x), 1000)
fit = fit_function(betaout, x_dummy)
fit_up = fit_function(betaout_up, x_dummy)
fit_dw = fit_function(betaout_dw, x_dummy)

plt.plot(x, y, 'o', color='#ff7f0e', label='Chebyshev')
plt.plot(x_dummy, fit, color='#ff7f0e')
axes.fill_between(x_dummy, fit_up, fit_dw, color='#ff7f0e', alpha=0.25)


text = (fr'$\alpha_\mathrm{{SOR}}={E1:0.3g}$ $\beta_\mathrm{{SOR}}={F1:0.3g}$' + '\n' + fr'$\alpha_\mathrm{{Cheb}}={E2:0.3g}$ $\beta_\mathrm{{Cheb}}={F2:0.3g}$')
boxprops = dict(boxstyle='round', facecolor='ivory', alpha=0.5)
axes.text(0.48, 0.2, text, transform=axes.transAxes, fontsize=14,
	verticalalignment='top', bbox=boxprops)


plt.legend(loc='best')
plt.savefig('images/alpah_fit.png')
plt.show()

