import numpy as np
from scipy import odr

import matplotlib.pyplot as plt
from matplotlib import rc

def FitterPlotter(fit_function, init_params, x, y, sx=None, sy=None, nstd=1.0,
	title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, datalabel='',
	showfig=True, color=None, savepath=None, textbox=None, figax=None,
	ret_fig=False):

	# Plotting setup
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)

	# Model function
	if fit_function == 'lin':
		def fit_function(beta, x):
			A, B = beta
			return A * x + B
	elif fit_function == 'exp':
		def fit_function(beta, x):
			A, B = beta
			return A * np.exp(B * x)
	elif not callable(fit_function):
		raise TypeError('fit_function is not callable.')

	# Turn fit_function into an odr model
	model = odr.Model(fit_function)

	# Create data object
	if sx == None and sy is None:
		data = odr.RealData(x, y)
	elif sx == None:
		data = odr.RealData(x, y, sy=sy)
	elif sy == None:
		data = odr.RealData(x, y, sx=sx)
	else:
		data = odr.RealData(x, y, sx=sx, sy=sy)

	# Set up ODR with model and data
	odrm = odr.ODR(data, model, beta0=init_params)

	# Run regression
	out = odrm.run()

	# Extract parameters
	betaout = out.beta
	betaerr = out.sd_beta
	print('Fit parameters and their std deviations')
	print('-------------------')
	for i in range(len(betaout)):
		print('Parameter {}: '.format(i + 1) + str(betaout[i])
			+ ' +- ' + str(betaerr[i]))

	# Fit curve and confidence intervals
	betaout_up = betaout + nstd * betaerr
	betaout_dw = betaout - nstd * betaerr

	x_dummy = np.linspace(min(x), max(x), 1000)
	fit = fit_function(betaout, x_dummy)
	fit_up = fit_function(betaout_up, x_dummy)
	fit_dw = fit_function(betaout_dw, x_dummy)

	# Plotting
	if figax is None: fig, ax = plt.subplots(1)
	else: fig, ax = figax

	if title is not None: ax.set_title(title)
	if xlabel is not None: ax.set_xlabel(xlabel)
	if ylabel is not None: ax.set_ylabel(ylabel)
	if xlim is not None: ax.set_xlim(xlim[0], xlim[1])
	if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
	ax.grid()

	if color is not None:
		points, = ax.plot(x, y, 'o', color=color, label=('Data ' + datalabel))
	else:
		points, = ax.plot(x, y, 'o', label=('Data ' + datalabel))
	ax.plot(x_dummy, fit, color=points.get_color())
	ax.fill_between(x_dummy, fit_up, fit_dw, color=points.get_color(),
		alpha=0.25, label='Confidence interval')

	# Plot textbox if necessary
	if textbox is not None:
		text = ''
		for i in textbox['params']:
			if textbox['error']:
				form = f'{int(np.log10(betaerr[i]))}' # WIP
				text += (textbox['symbols'][i] + fr'$={betaout[i]:0.3f}$' + '\n') #  \pm {betaerr[i]:0.3f}
		text = text[:-1]
		boxprops = dict(boxstyle='round', facecolor='ivory', alpha=0.5)
		ax.text(textbox['coords'][0], textbox['coords'][1], text,
			transform=ax.transAxes, fontsize=14, verticalalignment='top',
			bbox=boxprops)

	# Add legend, save and show
	ax.legend(loc='best')
	if savepath is not None: plt.savefig(savepath)
	if showfig: plt.show()

	# return figure and axes if necessary
	if ret_fig: return fig, ax
