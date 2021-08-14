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
x = np.log(np.arange(2, 20))
y = np.log(np.array([0.0004405699437484145, 0.0009890260407701135, 0.0032295179553329945, 0.017813001992180943, 0.047821735963225365, 0.09667080000508577, 0.20894004008732736, 0.46569363202434033, 0.8319363979389891, 1.489678509067744, 3.0394727259408683, 5.785604119068012, 12.890136440983042, 24.78353303601034, 37.575124494032934, 86.8564943370875, 157.48461006500293, 241.5062633859925]))

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

plt.xlabel(r'Number of Points $\ln(N)$')
plt.ylabel(r'Evaluation Time $\ln(t/[s])$')
plt.title(fr'Fitting the Evaluation Time')
plt.grid()

plt.plot(x, y, 'o', color='#1f77b4', label='Jacobi')
plt.plot(x_dummy, fit, color='#1f77b4')
axes.fill_between(x_dummy, fit_up, fit_dw, color='#1f77b4', alpha=0.25, label='Confidence interval')


#Data and data object
x = np.log(np.arange(2, 20))
y = np.log(np.array([0.00018200103659182787, 0.00016346503980457783, 0.00033103697933256626, 0.0006817419780418277, 0.001831674948334694, 0.0018613620195537806, 0.00273911003023386, 0.004544342984445393, 0.020729096955619752, 0.031037682900205255, 0.019161387928761542, 0.02131194097455591, 0.027132056071422994, 0.06219277600757778, 0.05546296399552375, 0.051155490917153656, 0.0780186610063538, 0.07823382399510592]))

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

plt.plot(x, y, 'o', color='#ff7f0e', label=r'$\texttt{scipy.linalg.eigh}$')
plt.plot(x_dummy, fit, color='#ff7f0e')
axes.fill_between(x_dummy, fit_up, fit_dw, color='#ff7f0e', alpha=0.25)


#Data and data object
x = np.log(np.arange(2, 20))
y = np.log(np.array([0.003232046030461788, 0.00844489794690162, 0.05248607904650271, 0.053439919953234494, 0.1388740569818765, 0.19855222490150481, 0.25416073005180806, 0.391926096053794, 1.0347737789852545, 1.4015960609540343, 1.645430043921806, 2.278135532978922, 2.5163359380094334, 6.416240025078878, 6.9385288540506735, 8.79441157891415, 10.412706230999902, 13.379818591987714]))

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

plt.plot(x, y, 'o', color='tab:green', label='Power Iteration')
plt.plot(x_dummy, fit, color='tab:green')
axes.fill_between(x_dummy, fit_up, fit_dw, color='tab:green', alpha=0.25)


"""
text = (fr'$\alpha_\mathrm{{SOR}}={E1:0.3g}$ $\beta_\mathrm{{SOR}}={F1:0.3g}$' + '\n' + fr'$\alpha_\mathrm{{Cheb}}={E2:0.3g}$ $\beta_\mathrm{{Cheb}}={F2:0.3g}$')
boxprops = dict(boxstyle='round', facecolor='ivory', alpha=0.5)
axes.text(0.48, 0.2, text, transform=axes.transAxes, fontsize=14,
	verticalalignment='top', bbox=boxprops)
"""

plt.legend(loc='best')
plt.savefig('images/eval_fit.png')
plt.show()

